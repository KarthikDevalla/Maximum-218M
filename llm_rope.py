import torch
torch.cuda.empty_cache()

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
with open("/teamspace/uploads/Harry_Potter_all_books_preprocessed.txt",'r',encoding='utf-8') as f1, open("/teamspace/uploads/blue train.txt") as f2, open("/teamspace/uploads/shakesphere.txt") as f3, open("/teamspace/uploads/murder.txt") as f4:
    text=f1.read()+f2.read()+f3.read()+f4.read()
tokenizer=tiktoken.get_encoding('gpt2')
encoding_values=tokenizer.encode(text)

print('Text Length:',len(text))
print(f'Number of Training Tokens: {len(encoding_values):,}')
GPT_CONFIG_219M={
    'vocab_size':50257,
    'vector_dim':768,
    'context_length':1024,
    'n_layers':20,
    'dropout_rate':0.2,
    'qkv_bias':False
}
print(GPT_CONFIG_219M)
train_ratio=0.80
train_set=text[:int(len(text)*train_ratio)]
test_set=text[int(len(text)*train_ratio):]

class load_data(Dataset): 
    def __init__(self,text,tokenizer,stride,context_length):
        self.input_ids=[]
        self.target_ids=[]

        encodings=tokenizer.encode(text)
        for i in range(0,len(encodings)-context_length,stride):
            input_chunk=encodings[i:i+context_length]
            target_chunk=encodings[i+1:i+context_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
        

def create_dataloader(text,batch_size,num_workers=0,context_length=256,stride=256,shuffle=True,drop_last=True):
    tokenizer=tiktoken.get_encoding('gpt2')
    dataset=load_data(text,tokenizer,stride,context_length)
    
    dataloader= DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader

train_loader=create_dataloader(train_set,
                               batch_size=2,
                               num_workers=0,
                               context_length=GPT_CONFIG_219M['context_length'],
                               stride=GPT_CONFIG_219M['context_length'],
                               shuffle=True,
                               drop_last=True)

validation_loader=create_dataloader(test_set,
                               batch_size=2,
                               num_workers=0,
                               context_length=GPT_CONFIG_219M['context_length'],
                               stride=GPT_CONFIG_219M['context_length'],
                               shuffle=True,
                               drop_last=True)

class MultiHeadAttentionWithRope(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,n_heads,qkv_bias=False):
        super().__init__()
        self.d_out=d_out
        self.n_heads=n_heads
        self.head_dim=d_out//self.n_heads
        self.proj=nn.Linear(d_out,d_out) 
        self.W_q=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.register_buffer('causal_mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.dropout_mask=nn.Dropout(dropout)
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)))

    
    def _compute_rope_embeddings(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
        
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
        
    def _apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos.view(1, 1, cos.shape[0], cos.shape[1]) 
        sin = sin.view(1, 1, sin.shape[0], sin.shape[1]) 
        return (x * cos) + (self._rotate_half(x) * sin)
      

    def forward(self,x):
        b, num_tokens, d_in =x.shape
        queries_matrix=self.W_q(x)
        keys_matrix=self.W_k(x)
        values_matrix=self.W_v(x)

        queries_matrix=queries_matrix.view(b,num_tokens,self.n_heads,self.head_dim)
        keys_matrix=keys_matrix.view(b,num_tokens,self.n_heads,self.head_dim)
        values_matrix=values_matrix.view(b,num_tokens,self.n_heads,self.head_dim)

        queries_matrix=queries_matrix.transpose(1,2) 
        keys_matrix=keys_matrix.transpose(1,2)
        values_matrix=values_matrix.transpose(1,2)

        cos, sin = self._compute_rope_embeddings(num_tokens)
    
        queries_matrix = self._apply_rotary_pos_emb(queries_matrix, cos, sin)
        keys_matrix = self._apply_rotary_pos_emb(keys_matrix, cos, sin)
        
        attention_scores=queries_matrix @ keys_matrix.transpose(2,3) 
        causal_attention_scores=attention_scores.masked_fill_(self.causal_mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        causal_attention_weights=self.dropout_mask(torch.softmax(causal_attention_scores/keys_matrix.shape[-1]**0.5,dim=-1))
        full_context_vectors=(causal_attention_weights @ values_matrix).transpose(1,2)
        full_context_vectors=full_context_vectors.contiguous().view(b,num_tokens,self.d_out)
        full_context_vectors=self.proj(full_context_vectors)
        return full_context_vectors

class LayerNorm(nn.Module):
    def __init__(self, vector_dim):
        super().__init__()
        self.eps=1e-5
        self.shift=nn.Parameter(torch.zeros(vector_dim))
        self.scale=nn.Parameter(torch.ones(vector_dim))
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        variance=x.var(dim=-1,keepdim=True, unbiased=False) 
        normalized_values=(x-mean)/torch.sqrt(variance+self.eps) 
        return self.scale*normalized_values+self.shift

class GeGLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x*torch.sigmoid(x)+ (0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*x**3))))
    

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(cfg['vector_dim'],cfg['vector_dim']*4),
                                   GeGLU(), 
       nn.Linear(cfg['vector_dim']*4,cfg['vector_dim'])
        
    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention=MultiHeadAttentionWithRope(d_in=cfg['vector_dim'],
                                          d_out=cfg['vector_dim'],
                                          context_length=cfg['context_length'],
                                          n_heads=cfg['n_heads'],
                                          dropout=cfg['dropout_rate'],
                                          qkv_bias=cfg['qkv_bias']
                                          )
        self.feed_forward=FeedForward(cfg)
        self.layer_norm_1=LayerNorm(cfg['vector_dim'])
        self.layer_norm_2=LayerNorm(cfg['vector_dim'])
        self.dropout=nn.Dropout(cfg['dropout_rate'])

    def forward(self,x):
        shortcut=x
        x=self.layer_norm_1(x)
        x=self.attention(x)
        x=self.dropout(x)
        x=x+shortcut

        shortcut=x
        x=self.layer_norm_2(x)
        x=self.feed_forward(x)
        x=self.dropout(x)
        x=x+shortcut
        return x


class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb=nn.Embedding(cfg['vocab_size'],cfg['vector_dim'])
        self.dropout=nn.Dropout(cfg['dropout_rate'])

        self.transformer_blocks=nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_layer_norm=LayerNorm(cfg['vector_dim'])
        self.out_head=nn.Linear(cfg['vector_dim'],cfg['vocab_size'],bias=False)
    
    def forward(self,x):
        b, seq_len=x.shape
        token_embedding_layer=self.token_emb(x)
        final_input_embeddings=self.dropout(token_embedding_layer)
        final_input_embeddings=self.transformer_blocks(final_input_embeddings)
        final_input_embeddings=self.final_layer_norm(final_input_embeddings)
        logits=self.out_head(final_input_embeddings)
        return logits

def generate_next_word(idx,max_new_tokens,model,context_length,temperature=0.0,top_k=None,eos_id=None):
    for _ in range(max_new_tokens):
        idx_content=idx[:,-context_length:]
        with torch.no_grad():
            logits=model(idx)
        last_row=logits[:,-1,:]

        if top_k is not None:
            top_logits,_=torch.topk(last_row,top_k)
            last_row=torch.where(last_row<top_logits[:,-1],torch.tensor(float('-inf')).to(last_row.device),last_row)

        if temperature>0.0:
            last_row=last_row/temperature
            probs=torch.softmax(last_row,dim=-1)
            idx_last=torch.multinomial(probs,num_samples=1)
        else:
            idx_last=torch.argmax(probs,dim=-1,keepdim=True)

        if idx_last==eos_id:
            break
        idx=torch.cat((idx,idx_last),dim=1)
    return idx

def calculate_batch_loss(input_batch,target_batch,model,device): 
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    logits=model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())

def calculate_loss(dataloader,model,device,num_batches=None):
    total_loss=0
    if dataloader is None:
        return 
    elif num_batches is None:
        num_batches=len(dataloader)
    else:
        num_batches=min(num_batches,len(dataloader))
    
    for i,(input_batch,target_batch) in enumerate(dataloader):
        if i< num_batches:
            loss=calculate_batch_loss(input_batch,target_batch,model,device)
            total_loss+=loss.item()
        else:
            break
    return total_loss/num_batches
    
def text_to_token(start_words,tokenizer):
    encoded_words=tokenizer.encode(start_words)
    return torch.tensor(encoded_words).unsqueeze(0)


def token_to_text(logits,tokenizer):
    decoded_words=logits.squeeze(0) 
    return tokenizer.decode(decoded_words.tolist())


def evaluate_model(train_loader, validation_loader,model,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss=calculate_loss(train_loader,model,device,num_batches=eval_iter)
        test_loss=calculate_loss(validation_loader,model,device,num_batches=eval_iter)
    model.train() 
    return train_loss,test_loss

def generate_text(model,tokenizer,start_words,device,temperature,top_k,eos_id,cfg):
    model.eval()
    input_tokens=text_to_token(start_words,tokenizer).to(device)
    context_length=cfg['context_length']
    with torch.no_grad():
        generated_tokens=generate_next_word(idx=input_tokens,
                                        model=model,context_length=context_length,max_new_tokens=50,temperature=temperature,top_k=top_k,eos_id=eos_id)

    decoded_words=token_to_text(generated_tokens,tokenizer)
    print('Generated words:',decoded_words.replace('\n',' '))
    model.train()


def pre_training(model,train_loader,validation_loader,epochs,device,optimizer,eval_freq,eval_iter,start_words,cfg,temperature=0.0,top_k=25,eos_id=None):
    train_loss_list,validation_loss_list=[],[]
    global_step=-1
    for epoch in range(epochs):
        model.train()
        for input_data, target_data in train_loader:
            optimizer.zero_grad() 
            loss=calculate_batch_loss(input_data,target_data,model,device) 
            loss.backward()
            optimizer.step() 
            global_step+=1

            
            if global_step % eval_freq==0:
                train_loss,valid_loss=evaluate_model(train_loader,validation_loader,model,device,eval_iter)
                train_loss_list.append(train_loss)
                validation_loss_list.append(valid_loss)
                print(f'Epoch-{epoch+1} Step-{global_step}====> Train loss: {train_loss:3f} Val loss: {valid_loss:3f}')
                

        generate_text(model,tokenizer,start_words,device,temperature,top_k,eos_id,cfg)
    return train_loss_list,validation_loss_list



torch.manual_seed(666)
import time
start=time.time()
model=GPT(GPT_CONFIG_219M)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epochs=10
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1,betas=(0.9,0.95),eps=1e-8)
start_words='Every effort moves you'
train_loss_list,validation_loss_list=pre_training(model,train_loader,
                                                  validation_loader,
                                                  epochs,device,optimizer,
                                                  eval_freq=5,eval_iter=5,
                                                  start_words=start_words,
                                                  temperature=1.4,top_k=25,cfg=GPT_CONFIG_219M)
end=time.time()
print(f'Time Taken to train the model : {(end-start)/60:2f}')
torch.save({'model_state':model.state_dict(),
            'optimizer_state':optimizer.state_dict()},'model_and_optimizer.pth')
import matplotlib.pyplot as plt
plt.plot(torch.arange(len(train_loss_list)),train_loss_list, label='Training loss')
plt.plot(torch.arange(len(train_loss_list)),validation_loss_list,label='Validation loss')
plt.legend()
plt.show()



hf_hub_download(repo_id="InHUMAN/Maximum-218M", filename="model_and_optimizer.pth",local_dir='/teamspace/studios/this_studio')

checkpoint=torch.load('/teamspace/studios/this_studio/model_and_optimizer.pth',map_location='cpu')
model.load_state_dict(checkpoint['model_state'])
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1,betas=(0.9,0.95),eps=1e-8)
optimizer.load_state_dict(checkpoint['optimizer_state'])
model.eval() 

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text="Is this what I have"
input_tokens=text_to_token(text,tokenizer)
with torch.no_grad():
    new_words=generate_next_word(input_tokens,60,model,1024,temperature=2,top_k=15)
decoded_text=token_to_text(new_words.flatten(),tokenizer)
print(decoded_text.replace('\n',' '))

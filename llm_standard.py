import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken
with open("full_gpt_training.txt",'r',encoding='utf-8') as f:
    text=f.read()
tokenizer=tiktoken.get_encoding('gpt2')
encoding_values=tokenizer.encode(text)

print('Text Length:',len(text))
print(f'Number of Training Tokens: {len(encoding_values):,}')
GPT_CONFIG_160M={
    'vocab_size':50257,
    'vector_dim':768,
    'context_length':256, #can use 1024 which gpt-2 original coontext length but i will take higher compute power so we will stcik with 256
    'n_heads':12,
    'n_layers':12,
    'dropout_rate':0.1,
    'qkv_bias':False
}
print(GPT_CONFIG_160M)
train_ratio=0.90
train_set=text[:int(len(text)*train_ratio)]
test_set=text[int(len(text)*train_ratio):]

#The amount of token the model should give attention to before predicting the next word is called context_length.
#the model looks at 4 tokens to predict the next token, you could size of training array
# Lets's assume input-->X=[1,2,3,4] and output-->y=[2,3,4,5]
# For input X=[1]-->y=[2]
# For input X=[1,2]-->y=[3]
# For input X=[1,2,3]-->y=[4]
# For input X=[1,2,3,4]-->y=[5]

#COMPLETE DATA PREPROCESSING PIPELINE
#=====================================================================
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
                               batch_size=4,
                               num_workers=0,
                               context_length=GPT_CONFIG_160M['context_length'],
                               stride=GPT_CONFIG_160M['context_length'],
                               shuffle=True,
                               drop_last=True)

validation_loader=create_dataloader(test_set,
                               batch_size=4,
                               num_workers=0,
                               context_length=GPT_CONFIG_160M['context_length'],
                               stride=GPT_CONFIG_160M['context_length'],
                               shuffle=True,
                               drop_last=True)


# MULTI-HEAD ATTENTION
#=====================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,n_heads,qkv_bias=False):
        super().__init__()
        self.d_out=d_out
        self.n_heads=n_heads
        self.head_dim=d_out//self.n_heads
        self.proj=nn.Linear(d_out,d_out) # layer to combine head ouputs
        self.W_q=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.register_buffer('causal_mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
        self.dropout_mask=nn.Dropout(dropout)

    def forward(self,x):
        b, num_tokens, d_in =x.shape
        queries_matrix=self.W_q(x)
        keys_matrix=self.W_k(x)
        values_matrix=self.W_v(x)
        # we unroll the last dimension to account for the heads(so that heads are included)
        # so this batch_shape will turn from (1,3,6) to (1,3,2,3) after this reshaping we get the matrix that looks somethig like this:
        #(1,3,2,3)
        # 1----> Batch_size
        # 3----> Num of tokens
        # 2----> num of heads
        # 3----> d_out
        # queries=tensor([[
            #token-1  <------- # [[2.3,4.5,6.7],---> First_head
                            # [1.2,0.6,8.9]],---> second_head            

            #token-2  <------- # [[2.8,4.3,9.6],
                            # [5.6,1.3,8.1]],

            #token-3  <------- # [[3.9,2.5,7.4],
                            # [5.5,1.9,3.6]]]])
                            
        #By doing the below operations we will have a tensor like above.
        queries_matrix=queries_matrix.view(b,num_tokens,self.n_heads,self.head_dim)
        keys_matrix=keys_matrix.view(b,num_tokens,self.n_heads,self.head_dim)
        values_matrix=values_matrix.view(b,num_tokens,self.n_heads,self.head_dim)

        # In the above example notice that we grouped based on the number of tokens, but now we will group based on heads
        # to do this we will transpose
        # Remember we had this
        #(1,3,2,3)
        # 1----> Batch_size
        # 3----> Num of tokens
        # 2----> num of heads
        # 3----> head_dim

        # we want this shape to be (1,2,3,3) from (1,3,2,3), as python follows 0 based indexing, we swap the 1st =index and 2nd index.
        # the reason we do this is we want to calculate attention scores for each head. 
        queries_matrix=queries_matrix.transpose(1,2) #---> why did we do .transpose(1,2), this is the swapping syntax.
        keys_matrix=keys_matrix.transpose(1,2)
        values_matrix=values_matrix.transpose(1,2)
        # the resultant matrix will look something like this:
        # notice that this is exactly the same output as simple attention implementation
        # queries=tensor([[
                        # this the queries matrix of first head
                        # [[2.3,4.5,6.7],-------
                        #  [2.8,4.3,9.6],       |---> head-1
                        #  [3.9,2.5,7.4]],-------
        
                        # this is the quries matrix of second head
                        # [[1.2,0.6,8.9],-------
                        #  [5.6,1.3,8.1],       |---->head-2
                        #  [5.5,1.9,3.6]]--------
        # ]]])
        # you'll get the same metrices for keys anda values also.
        attention_scores=queries_matrix @ keys_matrix.transpose(2,3) # we onlywant the tokens and vector dim
        causal_attention_scores=attention_scores.masked_fill_(self.causal_mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        causal_attention_weights=self.dropout_mask(torch.softmax(causal_attention_scores/keys_matrix.shape[-1]**0.5,dim=-1))
        # after calculating the context vectors we swap the indices again to get the num_heads and head_dim closer so that we can combine them to get d_out, earlier we split them like this and now we are joining them back
        full_context_vectors=(causal_attention_weights @ values_matrix).transpose(1,2)
        # and now we combine the last two dimensions
        full_context_vectors=full_context_vectors.contiguous().view(b,num_tokens,self.d_out)# this step will flatten the matrix and ypu will get the dimension as input_tokens*(d_in*n_heads)
        full_context_vectors=self.proj(full_context_vectors)# optional layer can be ignored but implemented in practice.

        return full_context_vectors
# LAYER NORMALIZATION
# ===============================================================================

class LayerNorm(nn.Module):
    def __init__(self, vector_dim):
        super().__init__()
        self.eps=1e-5
        # just additional params for smoother training
        self.shift=nn.Parameter(torch.zeros(vector_dim))
        self.scale=nn.Parameter(torch.ones(vector_dim))
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        variance=x.var(dim=-1,keepdim=True, unbiased=False) # related to bezels correction 
        #where we use n-1 instead of n, bu ti doesnt really matter as LLM have huge number of params and this precision is negligible.
        normalized_values=(x-mean)/torch.sqrt(variance+self.eps) # added a small number to prevent zero division error
        return self.scale*normalized_values+self.shift

# GeGLU ACTIVATION
# =================================================================================
# implementing the GeGLU class # Gated linear unit with GELU
class GeGLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x*torch.sigmoid(x)+ (0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*x**3))))
# GeLU ACTIVATION
#===================================================================================
# implementing the GeLU class
class GeLU(nn.Module): # Gaussian Error Linear Unit
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x+0.044715*x**3)))
    

# FEED FORWARD NEURAL NETWORK
#======================================================================================

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(cfg['vector_dim'],cfg['vector_dim']*4),# the second layer has 4 times more neurons that the input for expansion.
                                   GeGLU(), # using GeGLU ---->modified from GeLU
       nn.Linear(cfg['vector_dim']*4,cfg['vector_dim'])) # contraction
        
    def forward(self,x):
        return self.layers(x)

# TRANSFORMER BLOCK
#===========================================================================================
# implementing the backbone of LLM(transformer)

# Transformer Architecture

                # Shortcut
                #    |
                #    |
                # Layer Norm
                #    |
                #    |
                # Multi head attention
                #    |
                #    |
                # Dropout
                #    |
                #    |
                # Add short cut
                #    |
                #    |
                # shortcut
                #    |
                #    |
                # Layer Norm
                #    |
                #    |
                # Feed Forward
                #    |
                #    |
                # Dropout
                #    |
                #    |
                # Add short cut
                #    |
                #    |
                # output

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention=MultiHeadAttention(d_in=cfg['vector_dim'],
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

# THE GPT ARCHITECTURE
#==================================================================================
# the architecture

                # Input tokens----->'Every effort moves you'
                #      |
                #      |
                # token embedding layer
                #      |
                #      |
                # positional embedding layer
                #      |
                #      |
                # Dropout
                #      |
                #      |
                # Transformer block
                #      |
                #      |
                # Final layer norm
                #      |
                #      |
                # Linear output layer
                #      |
                #      |
                #   Output

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb=nn.Embedding(cfg['vocab_size'],cfg['vector_dim'])
        self.pos_emb=nn.Embedding(cfg['context_length'],cfg['vector_dim'])
        self.dropout=nn.Dropout(cfg['dropout_rate'])

        self.transformer_blocks=nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.final_layer_norm=LayerNorm(cfg['vector_dim'])
        self.out_head=nn.Linear(cfg['vector_dim'],cfg['vocab_size'],bias=False)
    
    def forward(self,x):
        b, seq_len=x.shape
        token_embedding_layer=self.token_emb(x)
        pos_embedding_layer=self.pos_emb(torch.arange(seq_len,device=x.device))
        final_input_embeddings=token_embedding_layer+pos_embedding_layer
        final_input_embeddings=self.dropout(final_input_embeddings)
        final_input_embeddings=self.transformer_blocks(final_input_embeddings)
        final_input_embeddings=self.final_layer_norm(final_input_embeddings)
        logits=self.out_head(final_input_embeddings)
        return logits

# GENRATING THE NEXT WORDS
#==========================================================================
# to predict the next word in the sequence we will follow this approach

                    # pull the last row of the logits for each batch
                    #            |
                    #            |
                    # apply softmax to normalize the logits
                    #            |
                    #            |
                    # identify the column with the largest value
                    #            |
                    #            |
                    # Decode it from the tokenizer 
                    #            |
                    #            |
                    # add the row back to the input
# In practice we repeat this process until it reaches a user specified number of generated tokens.

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
            probs=torch.softmax(last_row,dim=-1)#----------> Temperature scaling with topk sampling.
            idx_last=torch.multinomial(probs,num_samples=1)
        else:
            idx_last=torch.argmax(probs,dim=-1,keepdim=True)

        if idx_last==eos_id:
            break
        idx=torch.cat((idx,idx_last),dim=1)
    return idx

def calculate_batch_loss(input_batch,target_batch,model,device): # loss per batch
    input_batch,target_batch=input_batch.to(device),target_batch.to(device)
    logits=model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())

def calculate_loss(dataloader,model,device,num_batches=None): # total loss of the model
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
    
# PRE -TRAINING LOOP
#===================================================================================
def text_to_token(start_words,tokenizer):
    encoded_words=tokenizer.encode(start_words)
    return torch.tensor(encoded_words).unsqueeze(0) # added a batch dimension


def token_to_text(logits,tokenizer):
    decoded_words=logits.squeeze(0) # remove batch dimension
    return tokenizer.decode(decoded_words.tolist())


def evaluate_model(train_loader, validation_loader,model,device,eval_iter):
    model.eval() # setting the model to eval while evaluating losses and stuff
    with torch.no_grad():
        train_loss=calculate_loss(train_loader,model,device,num_batches=eval_iter)
        test_loss=calculate_loss(validation_loader,model,device,num_batches=eval_iter)
    model.train() # setting the model back to train after we are done calculating losses and stuff
    return train_loss,test_loss

def generate_text(model,tokenizer,start_words,device,temperature,top_k,eos_id):
    model.eval()
    input_tokens=text_to_token(start_words,tokenizer).to(device)
    context_length=model.pos_emb.weight.shape[0]
    with torch.no_grad():
        generated_tokens=generate_next_word(idx=input_tokens,
                                        model=model,context_length=context_length,max_new_tokens=50,temperature=temperature,top_k=top_k,eos_id=eos_id)

    decoded_words=token_to_text(generated_tokens,tokenizer)
    print('Generated words:',decoded_words.replace('\n',' '))
    model.train()


def pre_training(model,train_loader,validation_loader,epochs,device,optimizer,eval_freq,eval_iter,start_words,temperature=0.0,top_k=25,eos_id=None):
    train_loss_list,validation_loss_list=[],[]
    global_step=-1
    for epoch in range(epochs):
        model.train()
        for input_data, target_data in train_loader:
            optimizer.zero_grad() # zero the gradients for proper updates during training
            loss=calculate_batch_loss(input_data,target_data,model,device) # loss for one input and output.
            loss.backward()# calculate gradients
            optimizer.step() # update the gradients
            global_step+=1

            # this step is optional, but essential for checking if the model is learning.
            if global_step % eval_freq==0: # print the training loss and validation loss for every eval_freq: if eval_freq is 5 print it after every epochs
                train_loss,valid_loss=evaluate_model(train_loader,validation_loader,model,device,eval_iter)
                # you'll get the loss for eval_freq number of batches
                train_loss_list.append(train_loss)
                validation_loss_list.append(valid_loss)
                print(f'Epoch-{epoch+1} Step-{global_step}====> Train loss: {train_loss:3f} Val loss: {valid_loss:3f}')
                
        # printing the next tokens for every epoch to see if it makes sense.
        generate_text(model,tokenizer,start_words,device,temperature,top_k,eos_id)
    return train_loss_list,validation_loss_list


# PRE-TRAINING 
#=====================================================================================================
torch.manual_seed(666)
import time
start=time.time()
model=GPT(GPT_CONFIG_160M)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
epochs=3
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1,betas=(0.9,0.95),eps=1e-8,device=device)
start_words='Every effort moves you'
train_loss_list,validation_loss_list=pre_training(model,train_loader,
                                                  validation_loader,
                                                  epochs,device,optimizer,
                                                  eval_freq=5,eval_iter=5,
                                                  start_words=start_words,
                                                  temperature=2,top_k=25)
end=time.time()
print(f'Time Taken to train the model : {(end-start)/60:2f}')
torch.save({'model_state':model.state_dict(),
            'optimizer_state':optimizer.state_dict()},'model_and_optimizer.pth')
import matplotlib.pyplot as plt
plt.plot(torch.arange(len(train_loss_list)),train_loss_list, label='Training loss')
plt.plot(torch.arange(len(train_loss_list)),validation_loss_list,label='Validation loss')
plt.legend()
plt.show()


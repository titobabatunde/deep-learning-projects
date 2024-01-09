# Intro to Deep Learning 11-685 HW3P2
Click [here](https://www.kaggle.com/competitions/idl-hw4p2-slack/leaderboard) for kaggle link.


## Experimental Details
Click [here](https://wandb.ai/11685-cmu/hw4p2-ablations?workspace=user-titobabatunde) for wandb link <br />
1. Different architectures from start to finish with final validation distance of 15.505 <br /> before 60% of data was added. 

2. Shared Parameters:
    * CrossEntropyLoss(ignore_index=PAD_TOKEN) last two are mean while others are sum loss
    * ReduceLROnPlateau(mode='min', factor=0.5, patience=5, verbose=True)
    * Cepstral normalization is set to True
    * Although size of epochs vary between 100 and 150, early stopping was used
    * Dropout = 0.30 (Same for Time & Frequency Masking)

3. The main issue with my ablation study is that the training was very slow so less     <br /> time for altering different parameters and I kept getting cuda out of memory error

| <div style="width:290px">Model Highlights</div> | <div style="width:190px">Model Sizes</div> | <div style="width:190px">Teacher Forcing </div> | Batch Size | Weight Decay | Learning Rate | Valid distance | <div style="width:290px">Comments</div> | 
| :---------------- | :------: | :----: | :------: | :----: | :----: | :----: | :----: |
| Listener: mobilevnet embedding with output size listener_hidden_size, 3 bidirectional base_lstms, Encoder: tf_blocks=1, cdn_layers = 2, train-100 | embedding_size & encoder_hidden_size <br /> & listener_size =256, projection_size = 128, <br /> speller_size = 512, <br /> num_heads = 4 | epoch+1 %20==0 or epoch<80 and tf_rate >0.4 -> tf_rate -=0.10 | 64 | 5e-3 | 2e-4 | 20.68 | Took very long want to see tf_rate|
| Listener: mobilevnet embedding with output size listener_hidden_size, 3 bidirectional base_lstms, Encoder: tf_blocks=1, cdn_layers = 2, train-100 | embedding_size & encoder_hidden_size <br /> & listener_size = 96, projection_size = 42, <br /> speller_size = 192, <br /> num_heads = 4 | val_dist < 30 or epoch>20 and tf_rate >0.6 -> tf_rate *=0.985 | 64 | 5e-3 | 2e-4 | 47.94 | This is where I started getting cuda out of memory issue due to scaler and server issues|
| Listener: mobilevnet embedding with output size listener_hidden_size, 3 bidirectional base_lstms, Encoder: tf_blocks=1, cdn_layers = 2, train-100 | embedding_size & encoder_hidden_size <br /> & listener_size = 96, projection_size = 42, <br /> speller_size = 192, <br /> num_heads = 4 | epoch+1 %20==0 or epoch<80 and tf_rate >0.4 -> tf_rate -=0.10 | 64 | 5e-3 | 2e-4 | 67.36 | This didn't help|
| Listener: mobilevnet embedding with output size listener_hidden_size, 3 bidirectional base_lstms, Encoder: tf_blocks=1, cdn_layers = 2, train-100 | embedding_size & encoder_hidden_size <br /> & listener_size = 96, projection_size = 48, <br /> speller_size = 192, <br /> num_heads = 3 | epoch+1 %20==0 or epoch<80 and tf_rate >0.4 -> tf_rate -=0.10 | 64 | 5e-3 | 2e-4 | 430.63 | This also didn't help, oscillated about 434 dist|
| Listener: embedding with output size listener_hidden_size*2, 3 blocks of layers of pblstm, Encoder: tf_blocks=2, cdn_layers = 3, train-460 | embedding_size & encoder_hidden_size <br /> & listener_size & projection_size = 256, <br /> speller_size = 512, <br /> num_heads = 2 | val_dist < 30 or epoch>20 and tf_rate >0.6 -> tf_rate *=0.985 | 64 | 1e-6 | 1e-4 | 427.91 | Cuda out of memory |
| Listener: embedding with output size listener_hidden_size*2, 3 blocks of layers of pblstm, Encoder: tf_blocks=2, cdn_layers = 3, train-460 | embedding_size & encoder_hidden_size <br /> & listener_size & projection_size = 256, <br /> speller_size = 512, <br /> num_heads = 2 | val_dist < 30 or epoch>20 and tf_rate >0.6 -> tf_rate *=0.985 | 32 | 1e-6 | 1e-4 | 15.505 | Actually stopped this because it was taking too long |


## Running Instructions
python3 hw4p2_6_v2_posthackathon.py

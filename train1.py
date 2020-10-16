import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from evaluate_on_lfw import evaluate_LFW
import datetime
from tqdm.auto import tqdm
from datasets import TripletDataset
from network import *
import numpy as np
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
# Train model for MobileNet V2

def train():
    TRAIN_NAME = "MobileNetV2"
    log_dir = 'C:/tmp/facenet/logs'
    chk_dir = 'C:/tmp/facenet/checkpoints'
    # Create directories for training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_dir, TRAIN_NAME, current_time, 'train')
    os.path.join(log_dir, TRAIN_NAME, current_time, 'train')
    test_log_dir = os.path.join(log_dir, TRAIN_NAME, current_time, 'test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    ckpoint_dir = os.path.join(chk_dir, TRAIN_NAME, current_time)
    os.makedirs(ckpoint_dir, exist_ok=True)

    # Create the base model from the pre-trained model MobileNet V2
    model = get_MobileNet_backbone_network(embedding_size=128, fc_layer_size=1024, l2_norm=False, trainable_base=True)
    # model.load_weights('/HOME/FaceNet/checkpoints/ResNet50_no_l2/20200714-112711/best_epoch_92_weights.hdf5')
    #model.summary()
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    
    # Data loaders initialization
    data_dir = r'C:/bd_ai/dli/celeba/celeba_processed/'
    metadata_path = r'C:/bd_ai/dli/celeba/Anno/identity_CelebA_test.txt'
    #vgg_data_dir = r'/HOME/data/vggface2/processed'

    #alternate_training = True

    best_val_valid_pairs = 100

    #main_dataset = TripletDataset(64, 8)
    #main_dataset.load_vgg_face(vgg_data_dir)
    #main_dataset.split_train_val_data()
    #main_dataset.gen_tf_dataset()
    #cur_dataset = main_dataset

    #if alternate_training:
    alternate_dataset = TripletDataset(64, 4)
    alternate_dataset.load_celeb_a_dataset(data_dir, metadata_path)
    alternate_dataset.split_train_val_data()
    alternate_dataset.gen_tf_dataset()
    cur_dataset = alternate_dataset


    # Training loop
    n_epochs = 5
    
    for epoch in range(n_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_valid_pairs = tf.keras.metrics.Mean()
        progress = 0
        train_steps, val_steps = cur_dataset.get_steps_per_epoch()
        print('train_steps=', train_steps, 'val_steps=', val_steps)

        with tqdm(desc=f'Training - Epoch {epoch + 1}/{n_epochs}', unit=' imgs') as pbar:
            for X, y in cur_dataset.train_ds:
                loss, fraction = train_step(X, y, model, optimizer)
                epoch_loss_avg.update_state(loss)  # Add current batch loss
                epoch_valid_pairs.update_state(fraction)
                # Update progress bar
                progress += 1 / train_steps
                pbar.set_postfix(**{'Epoch loss': epoch_loss_avg.result().numpy(), 'Epoch valid pairs': epoch_valid_pairs.result().numpy(), 'Prctg of train set': progress})
                pbar.update(cur_dataset.get_batch_size())  # current batch size

        # End of epoch
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss_avg.result(), step=epoch)
            tf.summary.scalar('Valid training pairs in batch', epoch_valid_pairs.result(), step=epoch)

        val_loss_avg = tf.keras.metrics.Mean()
        val_valid_pairs = tf.keras.metrics.Mean()
        progress = 0
        with tqdm(desc=f'Validation - Epoch {epoch + 1}/{n_epochs}', unit=' imgs') as pbar:
            for X, y in cur_dataset.val_ds:
                loss, fraction = val_step(X, y, model)
                val_loss_avg.update_state(loss)  # Add current batch loss
                val_valid_pairs.update_state(fraction)
                # Update progress bar
                progress += 1 / val_steps
                pbar.set_postfix(**{'Prctg of validation set': progress})
                pbar.update(cur_dataset.get_batch_size())  # current batch size
        if (val_valid_pairs.result().numpy() < best_val_valid_pairs):
            print("New validation best result - Saving model")
            best_val_valid_pairs = val_valid_pairs.result().numpy()
            model.save_weights(os.path.join(ckpoint_dir, f"best_epoch_{epoch}_weights.hdf5"))

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss_avg.result(), step=epoch)
                tf.summary.scalar('Valid training pairs in batch', val_valid_pairs.result(), step=epoch)
            print("Epoch {:03d}: Train Loss: {:.3f} || Train valid pairs: {:.3f}% || Val Loss: {:.3f} || Val valid pairs: {:.3f}% ".format(epoch, epoch_loss_avg.result(), epoch_valid_pairs.result() * 100, val_loss_avg.result(), val_valid_pairs.result() * 100))
            # evaluate_LFW(model, embedding_size=128)

        #if alternate_training:
        #    if epoch % 2:
        #        cur_dataset = alternate_dataset
        #    else:
        #        cur_dataset = main_dataset

if __name__ == '__main__':
    train()
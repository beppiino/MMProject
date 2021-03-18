from keras.models import Model
from keras.layers import Input, Layer
from model import create_model, TripletLossLayer
from data import triplet_generator



def train_model():
    nn4_small2 = create_model()

    # Input for anchor, positive and negative images
    in_a = Input(shape=(96, 96, 3))
    in_p = Input(shape=(96, 96, 3))
    in_n = Input(shape=(96, 96, 3))

    # Output for anchor, positive and negative embedding vectors
    # The nn4_small model instance is shared (Siamese network)
    emb_a = nn4_small2(in_a)
    emb_p = nn4_small2(in_p)
    emb_n = nn4_small2(in_n)

    # Layer that computes the triplet loss from anchor, positive and negative embedding vectors
    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

    # Model that can be trained with anchor, positive negative images
    nn4_small2_train = Model([in_a, in_p, in_n], triplet_loss_layer)

    # triplet_generator() creates a generator that continuously returns
    # ([a_batch, p_batch, n_batch], None) tuples where a_batch, p_batch
    # and n_batch are batches of anchor, positive and negative RGB images
    # each having a shape of (batch_size, 96, 96, 3).
    generator = triplet_generator()

    nn4_small2_train.compile(loss=None, optimizer='adam')
    nn4_small2_train.fit_generator(generator, epochs=10, steps_per_epoch=100)

    return nn4_small2_train

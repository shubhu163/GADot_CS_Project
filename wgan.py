import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Define the Generator
def build_generator(input_dim, output_shape):
    noise_input = Input(shape=(input_dim,))
    x = Dense(128)(noise_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(output_shape), activation='tanh')(x)
    x = Reshape(output_shape)(x)
    generator = Model(noise_input, x, name='generator')
    return generator

# Define the Discriminator
def build_discriminator(input_shape):
    img_input = Input(shape=input_shape)
    x = Flatten()(img_input)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1)(x)
    discriminator = Model(img_input, x, name='discriminator')
    return discriminator

# Define the WGAN-GP model
class WGANGP:
    def __init__(self, generator, discriminator, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.compile(optimizer=Adam(0.000002, 0.5), loss=self.wasserstein_loss)
        self.discriminator.trainable = False
        noise = Input(shape=(latent_dim,))
        img = self.generator(noise)
        valid = self.discriminator(img)
        self.combined = Model(noise, valid)
        self.combined.compile(optimizer=Adam(0.000002, 0.5), loss=self.wasserstein_loss)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = tf.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        return tf.reduce_mean(gradient_penalty)

#     def train(self, X_train, epochs, batch_size, sample_interval=50):
#         valid = np.ones((batch_size, 1))
#         fake = -np.ones((batch_size, 1))
#         for epoch in range(epochs):
#             for _ in range(5):
#                 idx = np.random.randint(0, X_train.shape[0], batch_size)
#                 imgs = X_train[idx]
#                 noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#                 fake_imgs = self.generator.predict(noise)
#                 d_loss_real = self.discriminator.train_on_batch(imgs, valid)
#                 d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
#                 d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#             noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
#             g_loss = self.combined.train_on_batch(noise, valid)
#             print(f'Epoch: {epoch + 1}, D loss: {d_loss}, G loss: {g_loss}')

    def train(self, X_train, epochs, batch_size, sample_interval=50):
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        for epoch in range(epochs):
            for _ in range(5):  # Train the discriminator more frequently
                # Sample real and fake images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_imgs = self.generator.predict(noise)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)

                # Calculate the gradient penalty
                epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                interpolated_imgs = epsilon * imgs + (1 - epsilon) * fake_imgs
                with tf.GradientTape() as tape:
                    tape.watch(interpolated_imgs)
                    pred = self.discriminator(interpolated_imgs)
                grads = tape.gradient(pred, [interpolated_imgs])[0]
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)

                # Update the discriminator loss with the gradient penalty
                d_loss = d_loss_real + d_loss_fake + 10 * gradient_penalty  # The weight for gradient penalty is often set to 10

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print(f'Epoch: {epoch + 1}, D loss: {d_loss}, G loss: {g_loss}')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Load and preprocess the data
file_path = '/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/data/DATA/CICD2018_train.csv'
df = pd.read_csv(file_path)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
X = df.drop(' Label', axis=1).values
y = df[' Label'].apply(lambda x: 1 if x != 'Benign' else 0).values

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=SEED, stratify=y)

# Split the data into benign and DDoS samples
X_benign = X_train[y_train == 0]

# Reshape the data for the neural network
X_benign_reshaped = X_benign.reshape(-1, 77, 1, 1)  # Adjust the shape as needed

# Instantiate and train the WGAN-GP model
latent_dim = 100
generator = build_generator(latent_dim, (77, 1, 1))
discriminator = build_discriminator((77, 1, 1))
wgan_gp = WGANGP(generator, discriminator, (77, 1, 1), latent_dim)
wgan_gp.train(X_benign_reshaped, epochs=100, batch_size=32)
# Assume 'generator' is your trained generator model
generator.save('/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/Output_Models/train_generator_200.h5')


# Generate fake-benign samples

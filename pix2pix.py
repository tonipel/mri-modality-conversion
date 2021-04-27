import os
import time
import datetime
import argparse

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

BUFFER_SIZE = 8160
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100


def load_t1_to_t2(image_file):
    # Image files are combined images including T1 & T2 images
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    # input image is T1
    input_image = image[:, :w, :]
    input_image = tf.cast(input_image, tf.float32)

    # target image is T2
    target_image = image[:, w:, :]
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image


def load_t2_to_t1(image_file):
    # Image files are combined images including T1 & T2 images
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    # input image is T2
    input_image = image[:, w:, :]
    input_image = tf.cast(input_image, tf.float32)

    # target image is T1
    target_image = image[:, :w, :]
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image


def load_convert(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32)
    return image


def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(
        target_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, target_image


def resize_convert(input_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # target_image = tf.image.resize(target_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image


def random_crop(input_image, target_image):
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1
    target_image = (target_image / 127.5) - 1
    return input_image, target_image


def normalize_convert(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image


def random_jitter(input_image, target_image):
    # resizing to 286 x 286 x 3
    input_image, target_image = resize(input_image, target_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image


def load_image_train_t1_to_t2(image_file):
    input_image, real_image = load_t1_to_t2(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_train_t2_to_t1(image_file):
    input_image, real_image = load_t2_to_t1(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_convert(image_file):
    input_image = load_convert(image_file)
    input_image = resize_convert(input_image,  IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize_convert(input_image)

    return input_image


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)



def get_checkpoint_prefix():
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    return checkpoint_prefix


class Pix2pix(tf.Module):

    def __init__(self, epochs):
        self.epochs = epochs
        self.lambda_value = 100
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            2e-4, beta_1=0.5)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator)

    def discriminator_loss(self, disc_target_output, disc_generated_output):
        target_loss = self.loss_object(tf.ones_like(
            disc_target_output), disc_target_output)
        generated_loss = self.loss_object(tf.zeros_like(
            disc_generated_output), disc_generated_output)
        total_disc_loss = target_loss + generated_loss

        return total_disc_loss

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(
            disc_generated_output), disc_generated_output)
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.lambda_value * l1_loss)

        return total_gen_loss, gan_loss, l1_loss

    def train_step(self, input_image, target_image):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator(
                [input_image, target_image], training=True)
            disc_generated_output = self.discriminator(
                [input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_generated_output, gen_output, target_image)
            disc_loss = self.discriminator_loss(
                disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_total_loss, disc_loss

    def train(self, dataset, checkpoint_pr):
        print('Training started...')
        for epoch in range(self.epochs):
            start_time = time.time()
            for n, (input_image, target_image) in dataset.enumerate():
                gen_loss, disc_loss = self.train_step(
                    input_image, target_image)

            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=checkpoint_pr)

            t_sec = time.time()-start_time
            t_min, t_sec = divmod(t_sec, 60)
            t_hour, t_min = divmod(t_min, 60)

            print(f'--- Epoch {epoch+1} ---')
            print(
                f'Generator Loss: {gen_loss:.3f} | Discriminator Loss: {disc_loss:.3f}')
            print(f'Time: {t_hour:.0f}:{t_min:.0f}:{t_sec:.0f}\n')
        print('Training finished!')


def save_generated_images(model, test_input, name):
    prediction = model(test_input, training=True)
    fig = plt.figure()
    plt.imshow(prediction[0] * 0.5 + 0.5)

    plt.rcParams['savefig.facecolor'] = 'black'
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.axis('off')
    plt.savefig(name)


def main():

    parser = argparse.ArgumentParser(description='List of options')

    parser.add_argument('-f', dest='input',
                        help='Path to input folder', required=True)
    parser.add_argument('-m', dest='mode', help='Mode to run MRI image converison',
                        choices=['train', 'convert'], required=True)

    parser.add_argument('-c', dest='convert', help='Image translation direction',
                        choices=['t1_to_t2', 't2_to_t1'], required=True)

    parser.add_argument('-o', dest='output', help='Path to output folder')

    parser.add_argument('-e', dest='epochs',
                        help='Number of epochs for training')
    parser.add_argument('-s', dest='save', help='Path to save model')
    parser.add_argument('-l_t1_t2', dest='load_t1_t2',
                        help='Load saved model T1 to T2 from custom directory')
    parser.add_argument('-l_t2_t1', dest='load_t2_t1',
                        help='Load saved model T2 to T1 from custom directory')

    args = parser.parse_args()

    if args.mode == 'train':
        train_dataset = tf.data.Dataset.list_files(args.input + '*.png')
        if args.convert == 't1_to_t2':
            train_dataset = train_dataset.map(
                load_image_train_t1_to_t2, num_parallel_calls=tf.data.AUTOTUNE)
        elif args.convert == 't2_to_t1':
            train_dataset = train_dataset.map(
                load_image_train_t2_to_t1, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(BATCH_SIZE)

        model = Pix2pix(int(args.epochs))
        checkpoint_pr = get_checkpoint_prefix()
        model.train(train_dataset, checkpoint_pr)

        if args.convert == 't1_to_t2':
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
            tf.saved_model.save(model, args.save + 'T1_to_T2_model/')

        elif args.convert == 't2_to_t1':
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
            tf.saved_model.save(model, args.save + 'T2_to_T1_model/')

    elif args.mode == 'convert':

        filenames = []
        filelist = sorted(os.listdir(args.input),
                          key=lambda x: int(x.replace(".png", "")))

        for file in filelist:
            filenames.append(args.input + str(file))

        test_dataset_one = tf.data.Dataset.from_tensor_slices(filenames)
        test_dataset = test_dataset.map(load_image_convert)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        if args.convert == 't1_to_t2':
            saved_model = tf.saved_model.load(args.load_t1_t2)

        elif args.convert == 't2_to_t1':
            saved_model = tf.saved_model.load(args.load_t2_t1)

        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        i = 1

        for input_image in test_dataset:
            save_generated_images(saved_model.generator,
                                  input_image, (args.output + str(i) + '.png'))
            i += 1


if __name__ == "__main__":
    main()

from tensorflow import (random_normal_initializer,
                        GradientTape,
                        reduce_mean,
                        zeros_like,
                        ones_like,
                        function,
                        math,
                        abs)
from tensorflow.summary import (create_file_writer,
                                scalar)
from keras.layers import (BatchNormalization,
                          Conv2DTranspose,
                          ZeroPadding2D,
                          concatenate,
                          LeakyReLU,
                          Dropout,
                          Conv2D,
                          Input,
                          ReLU)
from keras.losses import BinaryCrossentropy
from tensorflow.train import Checkpoint
from keras.optimizers import Adam
from keras import (Sequential,
                   Model)
from os.path import join
from numpy import array
import time

loss_object = BinaryCrossentropy(from_logits=True)
OUTPUT_CHANNELS = 1
LAMBDA = 100


def upsample(filters,
             size,
             apply_dropout=False) -> Model:
    initializer = random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(
        Conv2DTranspose(filters,
                        size,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False)
    )
    result.add(BatchNormalization())
    if apply_dropout:
        pass
        result.add(Dropout(0.5))
    # result.add(ReLU())
    return result

def get_conv_layer(filters,
                   size) -> Model:
    initializer = random_normal_initializer(0, 0.02)
    result = Sequential([
        Conv2D(filters,
               size,
               padding="same",
               kernel_initializer=initializer,
               ),
        BatchNormalization(),
        # ReLU(),
    ])
    return result


def get_conv_blocks() -> list:
    conv_blocks = [
        # get_conv_layer(128, 3),
        # get_conv_layer(128, 3),
        # get_conv_layer(64, 3),
        get_conv_layer(64, 4),
        # get_conv_layer(32, 3),
        # get_conv_layer(32, 3),
        # get_conv_layer(128, 3),
    ]
    return conv_blocks



def downsample(filters,
               size,
               apply_batchnorm=True) -> Model:
    initializer = random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(
        Conv2D(filters,
               size,
               strides=2,
               padding='same',
               kernel_initializer=initializer,
               use_bias=False)
    )
    if apply_batchnorm:
        pass
        result.add(BatchNormalization())
    # result.add(LeakyReLU())
    return result


def Generator() -> Model:
    left_input = Input(shape=[256,
                              256,
                              3],
                       name="left_generator")
    right_input = Input(shape=[256,
                               256,
                               3],
                        name="right_generator")
    # conv_blocks = get_conv_blocks()
    down_stack = [
        # (batch_size, 128, 128, 64)
        downsample(32,
                   4),
        # (batch_size, 64, 64, 128)
        downsample(64,
                   4),
        # (batch_size, 32, 32, 256)
        downsample(128,
                   4),
        # downsample(256,
                  # 4),
    ]
    up_stack = [
        # upsample(512,
                 # 4),
        # (batch_size, 32, 32, 512)
        # upsample(256,
                 # 4),
        # (batch_size, 64, 64, 256)
        upsample(64,
                 4),
        # (batch_size, 128, 128, 128)
        upsample(32,
                 4),
    ]
    conv_blocks = get_conv_blocks()
    initializer = random_normal_initializer(0., 0.02)
    # (batch_size, 256, 256, 3)
    conv1 = Conv2D(64,
                  4,
                  padding="same")
    conv2 = Conv2D(32,
                   4,
                   padding="same")
    last = Conv2DTranspose(OUTPUT_CHANNELS,
                           4,
                           strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           activation="tanh")
    x1 = left_input
    x2 = right_input
    for conv_block in conv_blocks:
        x1 = conv_block(x1)
        x2 = conv_block(x2)
    x3 = math.subtract(x1,
                      x2)
    x4 = math.add(x1,
                  x2)
    x = concatenate([x3,
                     x4])
    # x2 = left_input
    # x2 = right_input
    # x = concatenate([x1,
                     # x2])
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concatenate([x, skip])
    x=conv1(x)
    x=conv2(x)
    x = last(x)
    return Model(inputs=[left_input,
                         right_input],
                 outputs=x)


def generator_loss(disc_generated_output,
                   gen_output, target):
    gan_loss = loss_object(ones_like(disc_generated_output),
                           disc_generated_output)
    # Mean absolute error
    l1_loss = reduce_mean(abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator() -> Model:
    initializer = random_normal_initializer(0., 0.02)
    left_input = Input(shape=[256, 256, 3],
                       name='left_input_image')
    right_input = Input(shape=[256, 256, 3],
                        name='right_input_image')
    # (batch_size, 256, 256, channels*2)
    tar = Input(shape=[256, 256, 1],
                name='target_image')
    # x = math.subtract(left_input,
                       # right_input)
    x = concatenate([left_input,
                     right_input,
                     tar])
    # x = concatenate([inp, tar])
    # (batch_size, 128, 128, 64)
    down1 = downsample(64, 4, False)(x)
    # (batch_size, 64, 64, 128)
    down2 = downsample(128, 4)(down1)
    # (batch_size, 32, 32, 256)
    # down3 = downsample(256, 4)(down2)
    # (batch_size, 34, 34, 256)
    zero_pad1 = ZeroPadding2D()(down2)
    # (batch_size, 31, 31, 512)
    conv = Conv2D(512,
                  4,
                  strides=1,
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1)
    batchnorm1 = BatchNormalization()(conv)
    # batchnorm1=conv
    leaky_relu = batchnorm1
    # leaky_relu = LeakyReLU()(batchnorm1)
    # (batch_size, 33, 33, 512)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    # (batch_size, 30, 30, 1)
    last = Conv2D(1,
                  4,
                  strides=1,
                  kernel_initializer=initializer)(zero_pad2)
    return Model(inputs=[left_input,
                         right_input,
                         tar],
                 outputs=last)


def discriminator_loss(disc_real_output,
                       disc_generated_output):
    real_loss = loss_object(ones_like(disc_real_output),
                            disc_real_output)
    generated_loss = loss_object(zeros_like(disc_generated_output),
                                 disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


class pix2pix:
    def __init__(self) -> None:
        self.checkpoint_path = "../Models/Checkpoint"
        self._create_model()

    def _create_model(self) -> None:
        self.discriminator = Discriminator()
        self.generator = Generator()
        print(self.generator.summary())
        print(self.discriminator.summary())
        self._get_optimizers()
        self._create_checkpoint()
        self._create_log_file()

    def _create_log_file(self) -> None:
        filename = "../model.log"
        self.summary_writer = create_file_writer(filename)

    def _get_optimizers(self) -> None:
        self.generator_optimizer = Adam(2e-4,
                                        beta_1=0.5)
        self.discriminator_optimizer = Adam(2e-4,
                                            beta_1=0.5)

    def _create_checkpoint(self) -> None:
        self.checkpoint = Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        self.checkpoint_prefix = join(self.checkpoint_path,
                                      "checkpoint_model_")

    def fit(self,
            train_dataset: list,
            steps: int):
        start = time.time()
        for step, data in train_dataset.repeat().take(steps).enumerate():
            left, right, target = data
            if (step) % 1000 == 0:
                if step != 0:
                    print(
                        '\nTime taken for 1000 steps: {:.2f} sec\n'.format(
                            time.time()-start
                        )
                    )
                start = time.time()
                print(f"Step: {step//1000}k")
            self._train_step(left,
                             right,
                             target,
                             step)
            # Training step
            if (step+1) % 10 == 0:
                print('.',
                      end='',
                      flush=True)
            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 10000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        self._save_models()

    @function
    def _train_step(self,
                    left_image: list,
                    right_image: list,
                    target: array,
                    step: int):
        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            gen_output = self.generator([left_image,
                                         right_image],
                                        training=True)
            disc_real_output = self.discriminator([left_image,
                                                   right_image,
                                                   target],
                                                  training=True)
            disc_generated_output = self.discriminator([left_image,
                                                        right_image,
                                                        gen_output],
                                                       training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output,
                gen_output,
                target
            )
            disc_loss = discriminator_loss(disc_real_output,
                                           disc_generated_output)
            generator_gradients = gen_tape.gradient(
                gen_total_loss,
                self.generator.trainable_variables
            )
            discriminator_gradients = disc_tape.gradient(
                disc_loss,
                self.discriminator.trainable_variables
            )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients,
                self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients,
                self.discriminator.trainable_variables)
        )
        with self.summary_writer.as_default():
            scalar('gen_total_loss',
                   gen_total_loss,
                   step=step//1000)
            scalar('gen_gan_loss',
                   gen_gan_loss,
                   step=step//1000)
            scalar('gen_l1_loss',
                   gen_l1_loss,
                   step=step//1000)
            scalar('disc_loss',
                   disc_loss,
                   step=step//1000)

    def _save_models(self) -> None:
        path = "../Models"
        filename = join(path,
                        "generator.h5")
        self.generator.save(filename)
        filename = join(path,
                        "discriminator.h5")
        self.discriminator.save(filename)

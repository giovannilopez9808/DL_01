from tensorflow import (random_normal_initializer,
                        GradientTape,
                        reduce_mean,
                        zeros_like,
                        ones_like,
                        function,
                        math,
                        abs)
from keras.layers import (BatchNormalization,
                          Conv2DTranspose,
                          ZeroPadding2D,
                          concatenate,
                          Conv2D,
                          Input)
from tensorflow.summary import (create_file_writer,
                                scalar)
from tensorflow.train import (latest_checkpoint,
                              Checkpoint)
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras import (Sequential,
                   Model)
from os.path import join
from numpy import array
import time

loss_object = BinaryCrossentropy(from_logits=True)
OUTPUT_CHANNELS = 1
LAMBDA = 100


def upsample(filters: int,
             size: int) -> Model:
    initializer = random_normal_initializer(0., 0.02)
    result = Sequential([
        Conv2DTranspose(filters,
                        size,
                        strides=2,
                        padding='same',
                        kernel_initializer=initializer,
                        use_bias=False),
        BatchNormalization()
    ])
    return result


def get_conv_layer(filters: int,
                   size: int) -> Model:
    initializer = random_normal_initializer(0, 0.02)
    result = Sequential([
        Conv2D(filters,
               size,
               padding="same",
               kernel_initializer=initializer,
               activation="tanh",
               ),
        BatchNormalization(),
    ])
    return result


def get_conv_blocks() -> list:
    conv_blocks = [
        get_conv_layer(64, 4),
    ]
    return conv_blocks


def downsample(filters: int,
               size: int) -> Model:
    initializer = random_normal_initializer(0., 0.02)
    result = Sequential([
        Conv2D(filters,
               size,
               strides=2,
               padding='same',
               kernel_initializer=initializer,
               use_bias=False),
        BatchNormalization()
    ])
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
    down_stack = [
        downsample(32,
                   4),
        downsample(64,
                   4),
        downsample(132,
                   4),
        # downsample(128,
        # 4),
    ]
    up_stack = [
        upsample(132,
                 4),
        upsample(64,
                 4),
        upsample(32,
                 4),
    ]
    conv_blocks = get_conv_blocks()
    initializer = random_normal_initializer(0., 0.02)
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
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concatenate([x, skip])
    x = conv1(x)
    x = conv2(x)
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
    target = Input(shape=[256, 256, 1],
                   name='target_image')
    zeropadding_1 = ZeroPadding2D()
    zeropadding_2 = ZeroPadding2D()
    conv_1 = Conv2D(512,
                    4,
                    strides=1,
                    kernel_initializer=initializer,
                    use_bias=False)
    conv_2 = Conv2D(1,
                    4,
                    strides=1,
                    kernel_initializer=initializer)
    x1 = math.subtract(left_input,
                       right_input)
    x2 = math.add(left_input,
                  right_input)
    x = concatenate([x1,
                     x2,
                     target])
    down1 = downsample(64, 4)(x)
    down2 = downsample(128, 4)(down1)
    zero_pad1 = zeropadding_1(down2)
    conv = conv_1(zero_pad1)
    batchnorm1 = BatchNormalization()(conv)
    zero_pad2 = zeropadding_2(batchnorm1)
    last = conv_2(zero_pad2)
    return Model(inputs=[left_input,
                         right_input,
                         target],
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
    def __init__(self,
                 params: dict) -> None:
        self.checkpoint_path = params["checkpoint path"]
        self._create_model()
        self._create_summary_file(params)

    def _create_summary_file(self,
                             params:dict)->None:
        path = params["log path"]
        train_path = f"{path}_train"
        val_path = f"{path}_val"
        test_path = f"{path}_test"
        self.train_summary = create_file_writer(train_path)
        self.val_summary = create_file_writer(val_path)
        self.test_summary = create_file_writer(test_path)
        

    def _create_model(self) -> None:
        self.discriminator = Discriminator()
        self.generator = Generator()
        self._get_optimizers()
        self._create_checkpoint()

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
            dataset,
            steps: int):
        start = time.time()
        for step, (train_data,val_data,test_data) in enumerate(
            zip(dataset.train.repeat().take(steps),
                dataset.val.repeat().take(steps),
                dataset.test.repeat().take(steps))):
            train_left, train_right, train_target = train_data
            val_left,val_right,val_target=val_data
            test_left,test_right,test_target=test_data
            if step%1000==0:
                print(f"Step: {step}k")
            self._train_step(train_left,
                             train_right,
                             train_target,
                             step)
            self._val_test_step(val_left,
                                val_right,
                                val_target,
                                step,
                                "Validation")
            self._val_test_step(test_left,
                                test_right,
                                test_target,
                                step,
                                "Test")
            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 20000 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    # @function
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
        results={
            "Total_loss": gen_total_loss,
            "Generator_GAN_loss": gen_gan_loss,
            "Gen_l1_loss": gen_l1_loss,
            "Discriminator_loss": disc_loss,
        }
        self._write_summary(results,
                            step,
                            "Train")

    def _val_test_step(self,
                       left_image: list,
                       right_image: list,
                       target: array,
                       step: int,
                       data:str):
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
        results={
            "Total_loss": gen_total_loss,
            "Generator_GAN_loss": gen_gan_loss,
            "Gen_l1_loss": gen_l1_loss,
            "Discriminator_loss": disc_loss,
        }
        self._write_summary(results,
                            step,
                            data)

    def restore(self) -> Model:
        latest = latest_checkpoint(self.checkpoint_path)
        self.checkpoint.restore(latest)

    def _write_summary(self,
                       results:dict,
                       step,
                       data:str)->None:
        if data=="Test":
            summary=self.test_summary
        if data=="Validation":
            summary=self.val_summary
        if data=="Train":
            summary=self.train_summary
        with summary.as_default():
            for key,value in results.items():
                scalar(key,
                       value,
                       step=step)

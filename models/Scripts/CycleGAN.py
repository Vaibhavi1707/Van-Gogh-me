import tensorflow as tf
from tensorflow import keras
from generator import make_generator
from discriminator import make_discriminator
from losses import generator_loss, disc_loss, similar_loss

class CycleGAN(keras.Model):
    def __init__(self, lambda_cycle):
        super(CycleGAN, self).__init__()
        
        self.gogh_generator = make_generator()
        self.gogh_discriminator = make_discriminator()

        self.celeb_generator = make_generator()
        self.celeb_discriminator = make_discriminator()
        
        self.lambda_cycle = lambda_cycle
        
    def compile(self):
        super(CycleGAN, self).compile()
        self.gogh_gen_opt = tf.keras.optimizers.Adam()
        self.celeb_gen_opt = tf.keras.optimizers.Adam()
        
        self.gogh_disc_opt = tf.keras.optimizers.Adam()
        self.celeb_disc_opt = tf.keras.optimizers.Adam()
        
        self.gen_loss_fn = generator_loss
        self.disc_loss_fn = disc_loss
        
        self.similar_loss_fn = similar_loss
        
    def train_step(self, batch_data):
        van_gogh, celeb = batch_data
        print(van_gogh.shape, celeb.shape)
        with tf.GradientTape(persistent = True) as tape:
            gogh_painting = self.gogh_generator(celeb, training = True)
            similar2celeb = self.celeb_generator(gogh_painting, training = True)
            
            celeb_photo_generated = self.celeb_generator(van_gogh, training = True)
            similar2gogh = self.gogh_generator(celeb_photo_generated, training = True)
            
            same_gogh = self.gogh_generator(van_gogh, training = True)
            same_celeb = self.celeb_generator(celeb, training = True)
            
            gogh_disc_real = self.gogh_discriminator(van_gogh, training = True)
            celeb_disc_real = self.celeb_discriminator(celeb, training = True)
            
            gogh_disc_fake = self.gogh_discriminator(van_gogh, training = True)
            celeb_disc_fake = self.celeb_discriminator(celeb, training = True)
            
            gogh_gen_loss = self.gen_loss_fn(gogh_disc_fake)
            celeb_gen_loss = self.gen_loss_fn(celeb_disc_fake)
            
            similar_loss = self.similar_loss_fn(celeb, similar2celeb, self.lambda_cycle) + self.similar_loss_fn(van_gogh, similar2gogh, self.lambda_cycle)
                
            total_gogh_loss = gogh_gen_loss + similar_loss + self.similar_loss_fn(van_gogh, same_gogh, self.lambda_cycle)
            total_celeb_loss = celeb_gen_loss + similar_loss + self.similar_loss_fn(celeb, same_celeb, self.lambda_cycle)
                
            gogh_disc_loss = self.disc_loss_fn(gogh_disc_real, gogh_disc_fake)
            celeb_disc_loss = self.disc_loss_fn(celeb_disc_real, celeb_disc_fake)
            
            # Gradient - todo
            gogh_gen_grad = tape.gradient(total_gogh_loss, self.gogh_generator.trainable_variables)
            celeb_gen_grad = tape.gradient(total_celeb_loss, self.celeb_generator.trainable_variables)
            
            gogh_disc_grad = tape.gradient(gogh_disc_loss, self.gogh_discriminator.trainable_variables)
            celeb_disc_grad = tape.gradient(celeb_disc_loss, self.celeb_discriminator.trainable_variables)
            
            self.gogh_gen_opt.apply_gradients(
                zip(gogh_gen_grad, self.gogh_generator.trainable_variables)
            )
            self.celeb_gen_opt.apply_gradients(
                zip(celeb_gen_grad, self.celeb_generator.trainable_variables)
            )
            
            self.gogh_disc_opt.apply_gradients(
                zip(gogh_disc_grad, self.gogh_discriminator.trainable_variables)
            )
            self.celeb_disc_opt.apply_gradients(
                zip(celeb_disc_grad, self.celeb_discriminator.trainable_variables)
            )
            
            return {
                "painting_gen_loss": total_gogh_loss,
                "celeb_gen_loss": total_celeb_loss,
                "gogh_disc_loss": gogh_disc_loss,
                "celeb_disc_loss": celeb_disc_loss
            }
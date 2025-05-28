# Import libraries
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Expand training data 
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'expanded_train_data/',
    target_size=(224, 224), 
    batch_size=32)

# Use transfer learning
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in conv_base.layers:
    layer.trainable = False
    
x = Flatten()(conv_base.output)
x = Dense(128, activation='relu')(x)  
x = Dense(3, activation='softmax')(x)

model = Model(inputs=conv_base.input, outputs=x)

# Compile and train
optimizer = Adam(lr=1e-5)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(patience=3)

model.fit(train_generator,
          epochs=30,
          validation_data=valid_generator,
          callbacks=[early_stop])
          
# Evaluate model          
loss, accuracy = model.evaluate(valid_generator)
print(f'Accuracy: {accuracy}')

# Make predictions
import numpy as np
from tensorflow.keras.preprocessing import image

test_img = image.load_img('test_images/hunger1.jpg', target_size=(224, 224))
test_img = image.img_to_array(test_img) / 255.0
test_img = np.expand_dims(test_img, axis=0)

classes = ['Healthy', 'Malnourished', 'Starving']

prediction = classes[np.argmax(model.predict(test_img))]
print(f'Predicted class: {prediction}')

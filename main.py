from ChaosGame import *
from ImageGeneration import *
from ModelFile import *
from matplotlib import pyplot as plt
import pylab
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import models

simulated = False
# Define variables for simulated dataset
num_of_seq = 3000           # Size of the dataset
random_elements = 'no'      # Number of random nucleotids in the dummy junction, either 'no', 'one' or 'multiple'
dummy_junction = 'GGTGAG'   # Sequence that simulates the juction (our identification target)
length = 80                 # Length of the simulated DNA sequences to analyse
dummy_nojunction = 'A'

# Define variables none real dataset
filepath_data = "splice_data.data"     # filepath to data
junction_type = 'EI'                   # choose the targeted data, either 'EI' or 'IE'

# Define variable k for number of nucleotids in one sequence
k = 4

# Path trained model
model_filepath = "model_EI_k4"

# Create simulated dataset

if simulated:
    dataset, dataset_label,error = create_dataset(dummy_junction,dummy_nojunction, length, num_of_seq, random_elements)
    print('\ndataset',dataset)
    print('dataset_label',dataset_label, ", error in label = ", error)

# Import real dataset
possible_junct = set(['EI', 'IE'])
junction_notype = (possible_junct - set([junction_type])).pop()

sequence = open(filepath_data)
seq = sequence.read()
seq_pre = seq.replace(" ", "").split("\n")
input_sequence = []
label_sequence = []
ei = 0
n = 0
for i in range(len(seq_pre) - 1):
    if ((seq_pre[i].split(",")[0]) == junction_type):
        ei = ei + 1
        input_sequence.append(seq_pre[i].split(",")[2])
        label_sequence.append(int("1"))
    elif ((seq_pre[i].split(",")[0]) == junction_notype):
        continue
    else:
        n = n + 1
        input_sequence.append(seq_pre[i].split(",")[2])
        label_sequence.append(int("0"))

print('junction samples =', ei, ' normal =', n)
label_sequence = np.array(label_sequence)

dataset = input_sequence[0:ei * 2]
dataset_label = label_sequence[0:ei * 2]

# Image representation
images = create_images(dataset,k)
images = images.transpose(0,2,3,1)
print(np.shape(images)) #the number of pixels is determined by the k

# Show images
for i in range(3):
    img = 0
    fig=plt.figure()
    #pylab.title('Chaos Game for k = {} mers'.format(k))
    for n in range (1,4):
      img = img +1
      fig.add_subplot(1,3,img)
      pylab.title('Channel {}'.format(n))
      pylab.imshow(images[i][:,:,n-1], interpolation='nearest')
      pylab.ylabel('Sequence {}'.format(i))
pylab.show()

# Get scaled test and train
x_train, y_train, x_test, y_test = train_test(images, dataset_label, k)
print(np.shape(x_train))
print(np.shape(x_test))

# Train the model (already trained models are provided)
cnn = model(k)
cnn.compile(optimizer = 'adam', loss='binary_crossentropy')
model_checkpoint = ModelCheckpoint(filepath="model", monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only = True)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.01, patience = 4, verbose = 1)

# If not trained model
# cnn.fit(x_train, y_train, batch_size = 36, epochs = 20, validation_split = 0.15, shuffle=True, callbacks = [early_stopping, model_checkpoint])

# Load train model
cnn = models.load_model(model_filepath)

# Final evaluation
y_true = y_test
y_pred = cnn.predict(x_test)
y_pred[y_pred>=0.5]=1
y_pred[y_pred<0.5]=0

cm = confusion_matrix(y_true, y_pred)

ev = evaluation(cm)
print('PERFORMANCE METRICS:', ev)
print('')
plot_confusion_matrix(y_true, y_pred, 'Test', np.array(["negative", "positive"]))
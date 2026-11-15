import os,sys,glob,re
import numpy as np 
import matplotlib.pyplot as plt


w_dir = '/Users/jonathan/Documents/applications/unet/output'


model_dir = f'model50'


train_file = os.path.join(w_dir, model_dir, 'train_dice.npy')
val_file = os.path.join(w_dir, model_dir, 'val_dice.npy')

train_dice = np.load(train_file)
val_dice = np.load(val_file)
# test_dice = np.loadtxt(test_file, delimiter=' ')


plt.plot(train_dice, label='train')
plt.plot(val_dice, label='val')

try:
    test_file = os.path.join(w_dir, model_dir, 'test_dice.npy')
    test_dice = np.load(test_file)
    plt.plot(np.arange(0,len(test_dice)*10,10), test_dice, label='test')
    # plt.plot(test_dice[:,0], test_dice[:,1])
except:
    print("No test dice file !")

plt.ylim([0.6,1])
plt.legend(loc='lower right')
plt.xlabel(r'epochs')
plt.savefig(w_dir + f'/{model_dir}/' + f'dice_score_scaled.png', dpi=200)
plt.show()
plt.close()

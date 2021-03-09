import os
import json
import matplotlib.pylab as plt

# with open('reward_epoch.json') as f:
#     d = json.load(f)
# print(d)
# lists = d.items() # sorted by key, return a list of tuples

# x, y = zip(*lists) # unpack a list of pairs into two tuples

# with plt.style.context('ggplot'):
#     plt.rcParams.update({'font.size': 15})
#     plt.figure()
#     plt.plot(x, y)
#     plt.xlabel('Training Epoch')
#     plt.ylabel('Reward')
#     plt.tight_layout()
#     path_fig = os.path.join('plots', 'plot_rewards.png')
#     plt.savefig(path_fig)
    
    
with open('action.json') as f:
    d_action = json.load(f)

epochs, actions = zip(*d_action.items()) # unpack a list of pairs into two tuples
print(actions[0])
for action in range(len(actions)):
    print(action)
a_0 = [action[0] for action in actions][:20]
a_1 = [action[1] for action in actions][:20]
a_2 = [action[2] for action in actions][:20]
a_3 = [action[3] for action in actions][:20]
epochs = epochs[:20]
       
with plt.style.context('ggplot'):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(epochs, a_0, marker='v', markersize=8, label = 'Coarse')
    plt.plot(epochs, a_1, marker='o', markersize=8, label = 'Fine 1')
    plt.plot(epochs, a_2, marker='^', markersize=8, label = 'Fine 2')
    plt.plot(epochs, a_3, marker='x', markersize=8, label = 'Fine 3')
    plt.xlabel('Training Epochs', fontsize=20)
    plt.ylabel('Exit Point', fontsize=20)
    leg = ax.legend(loc='upper right', ncol = 2, shadow = True, fancybox = True, fontsize = 18)
    leg.get_frame().set_alpha(0.5)
    plt.tight_layout()
    path_fig = os.path.join('plots', 'plot_actions.png')
    plt.savefig(path_fig)

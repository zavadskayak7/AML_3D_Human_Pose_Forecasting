import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
from utils import Prediction_Unnormalization, Prediction_Unnormalization_Test,mpjpe_error,all_actions


def seq_for_viz(path,skip,len_of_sequences_input,len_of_sequences_output,downsample_factor,joints_to_consider,subject_to_consider,
               actions_to_consider,number_of_sequences_to_visualize):
    
    #returns a dict with keys sub,action and values (N_to_vis,len_of_seqs,n_joints,3)
    
    len_of_all_sequences=len_of_sequences_input+len_of_sequences_output
    data = np.load(path+'data_3d_h36m.npz',allow_pickle=True)['positions_3d'].item()
    sub={}
    
    actions_to_consider==all_actions(actions_to_consider)
        
    for subject, actions in data.items():
        if subject in subject_to_consider:
            sub[subject]={}
        else:
            continue
        for action_name, positions in actions.items():
            if action_name.split(' ')[0] in actions_to_consider:
                new_positions=positions[::downsample_factor] # subsampling to 25 fps and skip 6 first frames
                #sub[subject][action_name.split(' ')[0]]=new_positions
                #print(subject,action_name)
                n_seq=int(math.ceil(new_positions.shape[0]- len_of_all_sequences+1))
                seq_data=np.zeros((n_seq,len_of_all_sequences,positions.shape[1],positions.shape[2]))
                i=0
                for idx in range(0,n_seq,skip):
                    curr_seq=new_positions [idx:idx+len_of_all_sequences]
                    seq_data[i,:,:,:]=curr_seq
                    i+=1

                sampled_sequence=seq_data[np.random.randint(seq_data.shape[0], size=number_of_sequences_to_visualize),:,:] # select random sequences
                if joints_to_consider==17:
                    joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
                    sampled_sequence=sampled_sequence[:,:,joints_to_keep,:]

                sub[subject][action_name.split(' ')[0]]=torch.Tensor(sampled_sequence)
            else:
                continue
                
    return sub



def update(num,data_gt,data_pred,plots_gt,plots_pred,joints_to_consider,fig,ax):
    # the code for connecting the coordinates with lines, to represent the human body, is taken from Julieta Martinez
    # https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
    
    gt_vals = data_gt[num]
    lcolor = "#8e8e8e"
    rcolor = "#383838"
    I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
    J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1

    if joints_to_consider==17:
        joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        joints_dict={ x:index for index, x in enumerate(joints_to_keep)}
        I=np.array([joints_dict[i] for i in I])
        J=np.array([joints_dict[i] for i in J])

    
    LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)


    for i in np.arange( len(I) ):
      x = np.array( [gt_vals[I[i], 0], gt_vals[J[i], 0]] )
      y = np.array( [gt_vals[I[i], 1], gt_vals[J[i], 1]] )
      z = np.array( [gt_vals[I[i], 2], gt_vals[J[i], 2]] )
      plots_gt[i][0].set_xdata(x)
      plots_gt[i][0].set_ydata(y)
      plots_gt[i][0].set_3d_properties(z)
      plots_gt[i][0].set_color(lcolor if LR[i] else rcolor)
    
    
    pred_vals = data_pred[num]
    lcolor = "#9b59b6"
    rcolor = "#2ecc71"
    

    for i in np.arange( len(I) ):
      x = np.array( [pred_vals[I[i], 0], pred_vals[J[i], 0]] )
      y = np.array( [pred_vals[I[i], 1], pred_vals[J[i], 1]] )
      z = np.array( [pred_vals[I[i], 2], pred_vals[J[i], 2]] )
      plots_pred[i][0].set_xdata(x)
      plots_pred[i][0].set_ydata(y)
      plots_pred[i][0].set_3d_properties(z)
      plots_pred[i][0].set_color(lcolor if LR[i] else rcolor)
    
    

    r = 0.5
    xroot, yroot, zroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2]
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred


def visualize_sequences(sub,len_of_sequences_input,len_of_sequences_output,joints_to_consider,subject_to_consider
                       ,actions_to_consider,number_of_sequences_to_visualize,model,A,normalize,train_stats):
    


    len_of_all_sequences=len_of_sequences_input+len_of_sequences_output
    
    if joints_to_consider==17:
        joints_to_keep=[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
        joints_dict={ x:index for index, x in enumerate(joints_to_keep)}
 
    actions_to_consider==all_actions(actions_to_consider)
      

             
    for subject in subject_to_consider:
        for action in actions_to_consider:
            for i in range(number_of_sequences_to_visualize):
                data_pred=sub[subject][action][i,0:len_of_sequences_input,:,:]
                data_gt=sub[subject][action][i,len_of_sequences_input:len_of_all_sequences,:,:]

                # prediction normalization to pass to model, gt stays original
                data_pred = data_pred.float()
                data_pred = Prediction_Unnormalization_Test(data_pred,normalize,train_stats,unnorm=False)

                data_pred=torch.unsqueeze(data_pred,0) # adds batch_diemsion=1
                #data_pred = data_pred.float()

                data_pred=model(data_pred.permute(0,3,1,2).cuda(),A)
                data_pred=torch.squeeze(data_pred,0).permute(0,2,1)
                data_pred=data_pred.cpu().data.numpy()

                loss=mpjpe_error(torch.Tensor(data_pred),torch.Tensor(data_gt),normalize,train_stats,vis=True)*1000

                # for plotting return data_pred to original coordinates
                data_pred = Prediction_Unnormalization_Test(data_pred,normalize,train_stats,unnorm=True)       
                
                

                assert data_pred.shape==data_gt.shape
                
                    # the code for connecting the coordinates with lines, to represent the human body, is taken from Julieta Martinez
                    # https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py

                fig = plt.figure()
                ax = Axes3D(fig)
                I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1
                J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1
                            #I=[joint for joint in I if joint in joints_to_keep]
                            #J=[joint for joint in J if joint in joints_to_keep]
                                # Left / right indicator
                if joints_to_consider==17:
                    I=np.array([joints_dict[i] for i in I])
                    J=np.array([joints_dict[i] for i in J])




                LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

                vals = np.zeros((joints_to_consider, 3))

                                # Make connection matrix
                plots_gt = []
                lcolor = "#8e8e8e"
                rcolor = "#383838"

                for i in np.arange( len(I) ):
                    x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
                    y = np.array( [vals[I[i], 1], vals[J[i], 1]] )
                    z = np.array( [vals[I[i], 2], vals[J[i], 2]] )
                    if i ==0:
                        plots_gt.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,label='GT'))
                    else:
                        plots_gt.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor))

                plots_pred = []
                lcolor = "#9b59b6"
                rcolor = "#2ecc71"

                for i in np.arange( len(I) ):
                    x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
                    y = np.array( [vals[I[i], 1], vals[J[i], 1]] )
                    z = np.array( [vals[I[i], 2], vals[J[i], 2]] )
                    if i ==0:
                        plots_pred.append(ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor,label='Pred'))
                    else:
                        plots_pred.append(ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor))


                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.legend(loc='lower left')






                ax.set_xlim3d([-1, 1.5])
                ax.set_xlabel('X')

                ax.set_ylim3d([-1, 1.5])
                ax.set_ylabel('Y')

                ax.set_zlim3d([0.0, 1.5])
                ax.set_zlabel('Z')
                ax.set_title(str(subject)+' '+str(action)+' for frames= '+str(len_of_sequences_output)+' loss in mm: '+str(loss.item()))

                #ax.set_title(str(subject)+' '+str(action)+' for frames= '+str(len_of_sequences_output))
                line_anim = animation.FuncAnimation(fig, update, len_of_sequences_output, fargs=(data_gt,data_pred,plots_gt,plots_pred,joints_to_consider
                                                                           ,fig,ax),
                                                   interval=70, blit=False)
                #line_anim.save('D:/data//'+str(action)+'.gif', writer='pillow')
                plt.show()
                #fig.canvas.draw()





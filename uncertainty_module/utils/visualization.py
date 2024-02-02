import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_rgb(rgb_image, viz_path, step_idx, ret=False):
    # Ensure the values are in the range [0, 1] (matplotlib requirement for RGB images)
    rgb_image = rgb_image / np.max(rgb_image)
    # Save the image
    if ret:
        return rgb_image
    save_path = os.path.join(viz_path, f'{step_idx}_pick_conf.png')
    plt.imsave(save_path, rgb_image.transpose(1,0,2))
    plt.close()

def visualize_pick_conf(pick_conf, viz_path, step_idx, ret=False):
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    pick_up_max = pick_conf[argmax]
    p0_pix = argmax[:2]
    p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])
    
    if ret:
        return pick_conf, p0_pix[::-1]
        
    plt.imshow(pick_conf.transpose(1,0,2), cmap='hot', interpolation='nearest')
    # Add p to the image
    # p_np = np.round(np.array(p[::-1]), 2)  # Reverse the array before rounding
    # plt.plot(p_np[1], p_np[0], 'g+', markersize=10, markeredgewidth=2)#, markeredgewidth=1, markeredgecolor='k')
    p0_pix_plot = p0_pix[::-1]
    plt.plot(p0_pix_plot[1], p0_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)
    

    plt.colorbar()
    save_path = os.path.join(viz_path, f'{step_idx}_pick_conf.png')
    plt.savefig(save_path)
    plt.close()
    
def visualize_place_conf(place_conf, viz_path, step_idx, ret=False):
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    place_conf_max = place_conf[argmax]
    p1_pix = argmax[:2]
    p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])
        
    place_conf_plot = place_conf[:,:,argmax[2]][:, :, np.newaxis] # (320, 160, 1)
    
    if ret:
        return place_conf_plot, p1_pix[::-1]
    
    plt.imshow(place_conf_plot.transpose(1,0,2), cmap='hot', interpolation='nearest')
    # Add p to the image
    # q_np = np.round(np.array(q[::-1]), 2)  # Reverse the array before rounding
    # plt.plot(q_np[1], q_np[0], 'g+', markersize=10, markeredgewidth=2)#, markeredgewidth=1, markeredgecolor='k')
    p1_pix_plot = p1_pix[::-1]
    plt.plot(p1_pix_plot[1], p1_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)
    

    plt.colorbar()
    save_path = os.path.join(viz_path, f'{step_idx}_place_conf.png')
    plt.savefig(save_path)
    plt.close()
    

def save_all_visualizations(lang_goal, rgb_img, pick_conf, place_conf, viz_path, step_idx):
    rgb = visualize_rgb(rgb_img, viz_path, step_idx, ret=True)
    pick_conf_plot, p0_pix_plot = visualize_pick_conf(pick_conf, viz_path, step_idx, ret=True)
    place_conf_plot, p1_pix_plot = visualize_place_conf(place_conf, viz_path, step_idx, ret=True)
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1)

    # Display the images on the subplots
    axs[0].set_title('lang:{}'.format(lang_goal))
    axs[0].imshow(rgb.transpose(1,0,2))
    im1 = axs[1].imshow(pick_conf_plot.transpose(1,0,2))
    im2 = axs[2].imshow(place_conf_plot.transpose(1,0,2))

    axs[1].plot(p0_pix_plot[1], p0_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)
    axs[2].plot(p1_pix_plot[1], p1_pix_plot[0], 'b+', markersize=10, markeredgewidth=2)

    # # Add colorbars to the subplots
    # fig.colorbar(im1, ax=axs[1])
    # fig.colorbar(im2, ax=axs[2])

    # Save the figure
    save_path = os.path.join(viz_path, f'{step_idx}_all_visualizations.png')
    print('save_path:', save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Close the figure
    plt.close()
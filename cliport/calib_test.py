"""Ravens main training script."""

import os
import pickle
import json

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment
from uncertainty_module.utils.visualization import visualize_rgb, visualize_pick_conf, visualize_place_conf, save_all_visualizations

@hydra.main(config_path='./cfg', config_name='calib_test')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])
    print('train configuration', vcfg['train_config'])
    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
    
    exp_name = vcfg['exp_name']
    uaa_name = 'uaa' if vcfg['action_selection']['enabled'] else 'base'
    attn_str = 'attn' if vcfg['action_selection']['attn_uaa'] else 'X'
    trans_str = 'trans' if vcfg['action_selection']['trans_uaa'] else 'X'
    name = '{}-{}-{}-{}-{}-{}-{}-{}-n{}'.format(exp_name,
                                                uaa_name,
                                                attn_str,
                                                trans_str, 
                                                vcfg['action_selection']['attn_tau'],
                                                vcfg['action_selection']['trans_tau'],
                                                eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            tcfg['action_selection'] = vcfg['action_selection'] # update action selection in tcfg
            
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)
            # import pdb; pdb.set_trace()
            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']
            
            visualization = vcfg['visualization']['enabled']
            save_failed = vcfg['visualization']['save_failed']
            


            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos):
                print(f'Test: {i + 1}/{n_demos}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)
                
                if visualization:
                    lang_goal_list = []
                    obs_list = []
                    pick_conf_list = []
                    place_conf_list = []
                    
                # set task
                if 'multi' in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f'Evaluating on {task_name}')
                else:
                    task_name = vcfg['eval_task']
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(task)
                obs = env.reset()
                info = env.info
                reward = 0

                # Start recording video (NOTE: super slow)
                if record:
                    video_name = f'{task_name}-{i+1:06d}'
                    if 'multi' in vcfg['model_task']:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                for _ in range(task.max_steps):
                    curr_obs = obs
                    act = agent.act(obs, info, goal)
                    lang_goal = info['lang_goal']
                    print(f'Lang Goal: {lang_goal}')
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                    if visualization:
                        # breakpoint()
                        lang_goal_list.append(lang_goal)
                        obs_list.append(agent.test_ds.get_image(curr_obs)[:,:,:3])
                        pick_conf_list.append(act['pick_conf'])
                        place_conf_list.append(act['place_conf'])
                    if done:
                        break
                
                if visualization:
                    # if not save_failed or (save_failed and 1-total_reward>10e-3):
                    failed_list = [10, 12, 33, 34, 44, 45, 76, 81, 84, 86]
                    if (i+1) in failed_list: 
                        viz_path = os.path.join(vcfg['visualization']['dir'], f'{name}')
                        step_reward = '{}-{:.2f}'.format(i+1, total_reward)
                        viz_path = os.path.join(viz_path, step_reward)
                        if not os.path.exists(viz_path):
                            os.makedirs(viz_path)
                        # breakpoint()
                        for step_idx, confs in enumerate(zip(obs_list, pick_conf_list, place_conf_list)):
                            rgb, pick_conf, place_conf = confs[0], confs[1], confs[2]
                            lang_goal = lang_goal_list[step_idx]
                            # visualize_pick_conf(pick_conf, viz_path, step_idx)
                            # visualize_place_conf(place_conf, viz_path, step_idx)
                            # breakpoint()
                            save_all_visualizations(lang_goal, rgb, pick_conf, place_conf, viz_path, step_idx)
                
                results.append((total_reward, info))
                mean_reward = np.mean([r for r, i in results])
                print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

                # End recording video
                if record:
                    env.end_rec()

            all_results[ckpt] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }

        # Save results in a json file.
        if vcfg['save_results']:
            print('save_json', save_json)
            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == '__main__':
    main()

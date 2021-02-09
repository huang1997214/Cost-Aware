import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="SAIRL")

    #General
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=0,
                        help="Number of baselines")

    #SAIRL
    parser.add_argument("--Epoch",
                        type=int,
                        default=5,
                        help="Training Epochs")
    parser.add_argument("--Lr_DDQN",
                        type=float,
                        default=0.001,
                        help="Dueling DQN Lr")
    parser.add_argument("--Lr_Dis",
                        type=float,
                        default=0.001,
                        help="Dis Lr")
    parser.add_argument("--SAIRL_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal SAIRL")
    parser.add_argument("--SAIRL_learn_interval",
                        type=int,
                        default=1,
                        help="SAIRL's learning interval")
    parser.add_argument("--SAIRL_update_freq",
                        type=int,
                        default=1200,
                        help="Update freq in Q-learning")
    parser.add_argument("--SAIRL_greedy",
                        type=int,
                        default=8,
                        help="Discover greedy")
    parser.add_argument("--sigma",
                        type=float,
                        default=0.01,
                        help="A weight to trade of")

    #DQN
    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal dqn")
    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="Dqn's learning interval")

    #VM Settings
    parser.add_argument("--VM_Type",
                        type=list,
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        help="VM Type")
    parser.add_argument("--VM_Cost",
                        type=list,
                        default=[1, 1, 2, 2, 4, 1, 1, 2, 2, 4],
                        help="VM Cost")
    parser.add_argument("--VM_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="VM Cpus")
    parser.add_argument("--VM_Num",
                        type=int,
                        default=10,
                        help="The number of VMs")
    parser.add_argument("--VM_capacity",
                        type=int,
                        default=1000,
                        help="VM capacity")

    #Job Settings
    parser.add_argument("--lamda",
                        type=int,
                        default=20,
                        help="The parameter used to control the length of each jobs.")
    parser.add_argument("--Job_Type",
                        type=float,
                        default=0.9,
                        help="The parameter used to control the type of each jobs.")
    parser.add_argument("--Job_Num",
                        type=int,
                        default=8000,
                        help="The number of jobs.")
    parser.add_argument("--Job_len_Mean",
                        type=int,
                        default=200,
                        help="The mean value of the normal distribution.")
    parser.add_argument("--Job_len_Std",
                        type=int,
                        default=20,
                        help="The std value of the normal distribution.")
    parser.add_argument("--Job_ddl",
                        type=float,
                        default=0.25,
                        help="Deadline time of each jobs")
    #Plot
    parser.add_argument("--Plot_labels",
                        type=list,
                        default=['b-', 'm-', 'g-', 'y-', 'r-', 'k-', 'w-'],
                        help="Deadline time of each jobs")
    return parser.parse_args()
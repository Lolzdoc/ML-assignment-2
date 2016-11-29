% Code may be changed in this script, but only where it states that it is allowed
% to do so
%
% To be clear, you may change any code in this entire project, but for the
% hand-in, you are to use the code as it is, and only change some code in
% valid places of this script. So if you feel experimental, which hopefully 
% you do, download two sets of the code and play around with one part as 
% you wish, and then use the original code (with some valid and necessary
% changes) in the hand-in
%
% Code part of ML-2016
%
% This script runs the game Snake (small version, where the length of the 
% snake doesn't increase when eating apples). There is no possibility to play 
% the game oneself, but it is possible to train an RL agent using tabular
% Q-learning, and to let an agent play the game.
%
% Bugs, ideas etcetera: send them to the course email

% Begin with a clean sheet
clc;
close all;
clearvars;

% Ensure same randomization process (repeatability)
rng(5);

% Specify number of non-terminal states and actions of the game; no need to
% explicitly store terminal states as their values are always equal to zero
nbr_states  = 4136;
nbr_actions = 3;   

% Define size of the snake grid (N-by-N)
N = 7;

% Define length of the snake (will be placed at center, pointing in
% random direction (north/east/south/west)
snake_len = 3;

% Define initial number of apples (placed at random locations, currently only tested with 1 apple)
nbr_apples = 1;

% Updates per second (when watching the agent play)
updates_per_sec = 16;                   % Allowed to be changed
pause_time      = 1 / updates_per_sec; % DO NOT CHANGE

% Set visualization settings (what you as programmer will see when the agent is playing)
show_fraction  = 1;                        % Allowed to be changed. 1: show everything, 0: show nothing, 0.1: show every tenth, and so on...
show_every_kth = round(1 / show_fraction); % DO NOT CHANGE

% Stuff related to learning agent (YOU SHOULD EXPERIMENT A LOT WITH THESE
% SETTINGS - SEE EXERCISE 7)
nbr_ep             = 5000;                                          % Number of episodes (full games until snake dies) to train
rewards            = struct('default', -5, 'apple', 1000, 'death', -100); % Experiment with different reward signals, to see which yield a good behaviour for the agent
gamm               = 00;                                           % Discount factor in Q-learning
alph               = 00;                                           % Learning rate in Q-learning
eps                = 0.00;                                           % Random action selection probability in epsilon-greedy Q-learning (lower: increase exploitation, higher: increase exploration)
alph_update_iter   = 0;                                             % 0: Never update alpha, Positive integer k: Update alpha every kth episode
alph_update_factor = 0.99;                                           % At alpha update: new alpha = old alpha * alph_update_factor
eps_update_iter    = 0;                                             % 0: Never update eps, Positive integer k: Update eps every kth episode
eps_update_factor  = 0.5;                                           % At eps update: new eps = old eps * eps_update_factor
Q_vals             = randn(nbr_states, nbr_actions);  

% Below two commands are useful when you have tranined your agent and later
% want to test it (see also Exercise 7). Remember to set alph = eps = 0 in 
% testing!

% save('Q_vals.mat', 'Q_vals');
 load 'temp.mat';             

% Set up state representations
try
    load states;
    disp('Successfully loaded states!');
catch
    % Beware: running this may take up to a few minutes (but it only needs
    % to be run once)
    disp('Getting state representation!');
    start_time = tic;
    states     = get_states(nbr_states, N);
    end_time   = toc(start_time);
    disp(['Done getting state representation! Elapsed time: ', num2str(end_time), ' seconds']);
    save('states.mat', 'states');
    disp('Successfully saved states!');
end

% Keep track of high score, minimum score and store all scores 
top_score  = 0;
min_score  = 500;
all_scores = nan(1, nbr_ep);

% This is the main loop for running the agent and/or learning process to play the game
for i = 1 : nbr_ep
    
    % Display what episode we're at and current weights
    disp(['EPISODE: ', num2str(i)]);
    
    % Check if learning rate and/or eps should decrease
    if rem(i, alph_update_iter) == 0
        disp('LOWERING ALPH!');
        alph = alph_update_factor * alph %#ok<*NOPTS>
    end
    if rem(i, eps_update_iter) == 0
        disp('LOWERING EPS!');
        eps = eps_update_factor * eps
    end
    
    % Generate initial snake grid and possibly show it
    close all;
    [grid, head_loc]         = gen_snake_grid(N, snake_len, nbr_apples); % Get initial stuff
    score                    = 0;                                        % Initial score: 0
    grid_show                = grid;                                     % What is shown on screen is different from what exact is happening "under the hood"
    grid_show(grid_show > 0) = 1;                                        % This is what is seen on screen
    if rem(i, show_every_kth) == 0
        figure; imagesc(grid_show)
    end

    % Run an episode of the game
    while 1

        % Get state information
        state     = grid_to_state_4_tuple(grid);
        state_idx = find(and(and(states(:, 1) == state(1), states(:, 2) == state(2)), ...
                             and(states(:, 3) == state(3), states(:, 4) == state(4))));
                           
        % epsilon-greedy action selection
        if rand < eps % Select random action
            action = randi(3);
        else % Select greedy action
            [~, action] = max(Q_vals(state_idx, :));
        end
        
        % Possibly pause for a while
        if rem(i, show_every_kth) == 0
            pause(pause_time);
        end

        % Update state
        [grid, score, reward, terminate] = update_snake_grid(grid, snake_len, score, rewards, action);

        % Check for termination
        if terminate

            % Compute terminal TD(1)-error
            target = reward; % No one-step lookahead here - simply look at the reward
            Q_val  = Q_vals(state_idx, action);
            pred   = Q_val; 
            td_err = target - pred;
            
            % Update Q-value based on TD(1)-error
            Q_vals(state_idx, action) = Q_val + alph * td_err;
            
            % Insert score into container
            all_scores(i) = score;
            
            % Display stuff
            disp(['GAME OVER! SCORE:       ', num2str(score)]);
            disp(['AVERAGE SCORE SO FAR:   ', num2str(mean(all_scores(1 : i)))]);
            if i >= 10
                disp(['AVERAGE SCORE LAST 10:  ', num2str(mean(all_scores(i - 9 : i)))]);
            end
            if i >= 100
                disp(['AVERAGE SCORE LAST 100: ', num2str(mean(all_scores(i - 99 : i)))]);
            end
            if score > top_score
            disp(['NEW HIGH SCORE!         ', num2str(score)]);
                top_score = score;
            end
            if score < min_score
            disp(['NEW SMALLEST SCORE!     ', num2str(score)]);
                min_score = score;
            end
            
            % Terminate
            break;
        end

        % Update what to show on screen
        grid_show                = grid;
        grid_show(grid_show > 0) = 1;
        if rem(i, show_every_kth) == 0
            imagesc(grid_show);
        end
        
        % Compute TD(1)-error
        next_state     = grid_to_state_4_tuple(grid);
        next_state_idx = find(and(and(states(:, 1) == next_state(1), states(:, 2) == next_state(2)), ...
                                  and(states(:, 3) == next_state(3), states(:, 4) == next_state(4))));
        target         = reward + gamm * max(Q_vals(next_state_idx, :));
        Q_val          = Q_vals(state_idx, action);
        pred           = Q_val; 
        td_err         = target - pred;

        % Update Q-value based on TD(1)-error
        Q_vals(state_idx, action) = Q_val + alph * td_err;
    end
end
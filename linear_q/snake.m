% Code may be changed in this script, but only where it states that it is allowed 
% to do so
%
% To be clear, you may change any code in this entire project, but for the 
% hand-in, you are to use the code as it is, and only change some code in
% valid places of this script (and some other functions). So if you feel 
% experimental, which hopefully you do, download two sets of the code 
% and play around with one part as you wish, and then use the original code
% (with some valid and necessary changes) in the hand-in
%
% Code part of ML-2016
%
% This script runs the game Snake. There is no possibility to play the game
% oneself, but it is possible to train (Q-learning with linear function 
% approximation) a policy used by an agent that can then play the game
%
% SEE extract_state_action_features.m IN WHICH YOU NEED TO ENGINEER 
% STATE-ACTION FEATURES (SEE EXERCISE 8)
%
% Bugs, ideas etcetera: send them to the course email

% Begin with a clean sheet
clc;
close all;
clearvars;

% Ensure same randomization process (repeatability)
rng(5);

% Define size of the snake grid (N-by-N)
N = 30;

% Define length of initial snake (will be placed at center, pointing in
% random direction (north/east/south/west))
snake_len_init = 10;

% Define initial number of apples (placed at random locations, currently only tested with 1 apple)
nbr_apples = 1;

% Updates per second
updates_per_sec = 20;                  % Allowed to be changed (though your code must handle 20 updates per second at the lowest)
pause_time      = 1 / updates_per_sec; % DO NOT CHANGE

% Set visualization settings (what you as programmer will see when the agent is playing)
show_fraction  = 1;                        % Allowed to be changed. 1: show everything, 0: show nothing, 0.1: show every tenth, and so on...
show_every_kth = round(1 / show_fraction); % DO NOT CHANGE

% Stuff related to learning agent (YOU SHOULD EXPERIMENT A LOT WITH THESE
% SETTINGS - SEE EXERCISE 8)
nbr_feats          = 3;                                             % Number of state-action features per action
nbr_ep             = 5000;                                          % Number of episodes (full games until snake dies) to train
rewards            = struct('default', 0, 'apple', 1, 'death', -1); % Experiment with different reward signals, to see which yield a good behaviour for the agent
gamm               = 0.9;                                           % Discount factor in Q-learning
alph               = 0.01;                                          % Learning rate in Q-learning
eps                = 0.01;                                          % Random action selection probability in epsilon-greedy Q-learning (lower: increase exploitation, higher: increase exploration)
alph_update_iter   = 0;                                             % 0: Never update alpha, Positive integer k: Update alpha every kth episode
alph_update_factor = 0.5;                                           % At alpha update: new alpha = old alpha * alph_update_factor
eps_update_iter    = 0;                                             % 0: Never update eps, Positive integer k: Update eps every kth episode
eps_update_factor  = 0.5;                                           % At eps update: new eps = old eps * eps_update_factor
weights            = randn(nbr_feats, 1);                           % I.i.d. N(0,1) initial weights

% Below two commands are useful when you have tranined your agent and later
% want to test it (see also Exercise 8). Remember to set alph = eps = 0 in 
% testing!

% save('weights.mat', 'weights');
% load weights;         

% Keep track of high score, minimum score and store all scores 
top_score  = 0;
min_score  = 500;
all_scores = nan(1, nbr_ep);

% This is the main loop for running the agent and/or learning process to play the game
for i = 1 : nbr_ep
    
    % Display what episode we're at and current weights
    disp(['EPISODE: ', num2str(i)]);
    disp('WEIGHTS: ')
    disp(weights)
    
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
    snake_len                          = snake_len_init;
    [grid, head_loc]                   = gen_snake_grid(N, snake_len, nbr_apples); % Get initial stuff
    score                              = 0;                                        % Initial score: 0
    [prev_head_loc_m, prev_head_loc_n] = find(grid == snake_len);                  % Used in updates
    prev_head_loc                      = [prev_head_loc_m, prev_head_loc_n];       % Used by player/agent
    prev_head_loc_agent                = prev_head_loc;                            % Agent also knows "previous" head location initially
    grid_show                          = grid;                                     % What is shown on screen is different from what exact is happening "under the hood"
    grid_show(grid_show > 0)           = 1;                                        % This is what is seen by the player
    prev_grid_show                     = grid_show;                                % Used by agent - needs at least two frames to keep track of head location
    if rem(i, show_every_kth) == 0
        figure; imagesc(grid_show)
    end

    % Initialize time-step
    t = 0;
    
    % This while-loop runs until the snake dies and the game ends
    while 1
        
        % Run epsilon-greedy Q-learning

        % Extract state-action features and update relevant stuff (prev_grid_show, prev_head_loc)
        % Note that once we begin computing TD(1)-errors, we make a one-step lookahead; therefore, 
        % for t > 0, we can simply copy the relevant things computed during the lookahead
        if t == 0
            [state_action_feats, prev_grid_show, prev_head_loc_agent] = extract_state_action_features(prev_grid_show, grid_show, prev_head_loc_agent, nbr_feats);
        else
            state_action_feats  = state_action_feats_future;
            prev_grid_show      = prev_grid_show_future;
            prev_head_loc_agent = prev_head_loc_agent_future; 
        end

        % epsilon-greedy action selection
        if rand < eps % Select random action
            action = randi(3);
        else % Select greedy action (maximizing Q(s,a))
            Q_vals      = Q_fun(weights, state_action_feats);
            [~, action] = max(Q_vals);
        end
        
        % Possibly pause for a while
        if rem(i, show_every_kth) == 0
            pause(pause_time);
        end
        
        % Update state and obtain reward 
        [grid, head_loc, prev_head_loc, snake_len, score, reward, terminate] = update_snake_grid(grid, head_loc, prev_head_loc, snake_len, score, rewards, action);
        
        % Check for termination
        if terminate
            
            % Compute terminal TD(1)-error
            target = reward; % No one-step lookahead here - simply look at the reward
            pred   = Q_fun(weights, state_action_feats, action); 
            td_err = target - pred;

            % Update weights based on TD(1)-error
            weights = weights + alph * td_err * state_action_feats(:, action);
            
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
        
        % Update what to show to the agent (and possibly programmer)
        grid_show                = grid;
        grid_show(grid_show > 0) = 1;
        if rem(i, show_every_kth) == 0
            imagesc(grid_show);
        end
        
        % Compute TD(1)-error
        [state_action_feats_future, prev_grid_show_future, prev_head_loc_agent_future] = extract_state_action_features(prev_grid_show, grid_show, prev_head_loc_agent, nbr_feats);
        target                                                                         = reward + gamm * max(Q_fun(weights, state_action_feats_future));
        pred                                                                           = Q_fun(weights, state_action_feats, action);
        td_err                                                                         = target - pred;

        % Update weights based on TD(1)-error
        weights = weights + alph * td_err * state_action_feats(:, action);
        
        % Update time-step
        t = t + 1;
    end
end
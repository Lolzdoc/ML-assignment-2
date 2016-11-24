function [values, policy, nbr_pol_iter] = policy_iteration(pol_eval_tol, next_state_idxs, rewards, gamm)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so. You need to implement policy iteration in this function
%
% Code part of ML-2016
%
% Function to run policy iteration to learn an optimal policy. Note that
% this implementation assumes eating apple is also a terminal state (and 
% not only hitting a wall / the body of the snake). Think about why this 
% is OK, i.e., why it will give rise to an optimal policy also for the 
% Snake game in which eating an apple is non-terminal (this may help you
% with exercise 4).
%
% Input:
%
% pol_eval_tol    - Policy evaluation stopping tolerance 
% next_state_idxs - nbr_states-by-nbr_actions matrix; each entry of this 
%                   matrix is an integer in {-1, 0, 1, 2, ..., nbr_states}.
%                   In particular, the ith row of next_state_idxs gives
%                   the state indexes of taking the left, forward and right 
%                   actions. The only exceptions to this is if an action leads 
%                   to any terminal state; if an action leads to death, then
%                   the corresponding entry in next_state_idxs is 0; if an
%                   action leads to eating an apple, then the corresponding
%                   entry in next_state_idxs is -1
% rewards         - Struct of the form struct('default', x, 'apple', y, 'death', z)
%                   Here x refers to the default reward, which is received
%                   when the snake only moves without dying and without
%                   eating an apple; y refers to the reward obtained when
%                   eating an apple; and z refers to the reward obtained when
%                   dying
% gamm            - Discount factor in [0,1]
%
% Output:
%
% values       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal values for all the non-terminal 
%                states
% policy       - 1-by-nbr_states vector; will after successful policy
%                iteration contain optimal actions to take for all the 
%                non-terminal states
% nbr_pol_iter - The number of iterations that the policy iteration runs
%                for; may be used e.g. for diagnostic purposes
%                   
% Bugs, ideas etcetera: send them to the course email

% Get number of non-terminal states and actions
[nbr_states, nbr_actions] = size(next_state_idxs);

% Arbitrary initialization of values and policy
values = randn(1, nbr_states);  
policy = randi(3, 1, nbr_states); % policy is size 1-by-nbr_states
                                  % the entries of policy are 1, 2 or 3
                                  % selected uniformly at random

% Counter over number of policy iterations, for possible diagnostic purposes
nbr_pol_iter = 0;

% This while-loop runs the policy iteration
while 1
    
    % Policy evaluation
    while 1
        
        Delta = 0;
        for state_idx = 1 : nbr_states
            current_state = state_idx;
            sum = 0;
            for action = 1:nbr_actions
                if (action == policy(current_state))
                    pol = 1;
                else
                    pol = 0;
                end
                next_state = next_state_idxs(current_state,action);
                if(next_state == 0)
                    sum = sum + pol * (1 * (rewards.default + gamm * rewards.death));
                elseif (next_state == -1)
                    sum = sum + pol * (1 * (rewards.default + gamm * rewards.apple));
                else
                    sum = sum + pol * (1 * (rewards.default + gamm * values(next_state)));
                end
                
            end
            
            if(abs(sum - values(current_state)) > Delta)
                Delta = abs(sum - values(current_state));
            end
            values(current_state) = sum;

            % FILL IN POLICY EVALUATION WITHIN THIS LOOP
        end
        
        % Check for policy evaluation termination
        if Delta < pol_eval_tol
            %disp(['Last Delta: ', num2str(Delta)])
            break;
        else
            disp(['Delta: ', num2str(Delta)])
        end
    end
    
    % Policy improvement
    policy_stable = true; 
    for state_idx = 1 : nbr_states
        % FILL IN POLICY IMPROVEMENT WITHIN THIS LOOP
        current_state = state_idx;
        current_max = 0;
        current_action = policy(current_state);
        
        for action = 1:nbr_actions
            sum = 0;
            next_state = next_state_idxs(current_state,action);
            if(next_state == 0)
                sum = sum + (1 * (rewards.default + gamm * rewards.death));
            elseif (next_state == -1)
                sum = sum + (1 * (rewards.default + gamm * rewards.apple));
            else
                sum = sum +(1 * (rewards.default + gamm * values(next_state)));
            end
            if(sum > current_max)
                current_max = sum;
                current_action = action;
            end
        end
        if (policy(current_state) ~= current_action)
            %disp(['current_state policy: ', num2str(policy(current_state)),'new policy', num2str(current_action)])
            policy_stable = false;
        end
        policy(current_state) = current_action;
    end
    
    % Increase the number of policy iterations 
    nbr_pol_iter = nbr_pol_iter + 1;
    
    % Check for policy iteration termination (terminate if and only if the
    % policy is no longer changing, i.e. if and only if the policy is stable)
    if policy_stable
        break;
    end
end
end
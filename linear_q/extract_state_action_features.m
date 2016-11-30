function [state_action_feats, prev_grid, prev_head_loc] = extract_state_action_features(prev_grid, grid, prev_head_loc, nbr_feats)
%
% Code may be changed in this function, but only where it states that it is 
% allowed to do so
%
% Code part of ML-2016
%
% Function to extract state-action features, based on current and previous
% grids (game screens)
%
% Input:
%
% prev_grid     - Previous grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% grid          - Current grid (game screen), N-by-N matrix. If initial
%                 time-step: prev_grid = grid, else: prev_grid != grid.
% prev_head_loc - The previous location of the head of the snake (from the 
%                 previous time-step). If initial time-step: Assumed known,
%                 else: inferred in function "update_snake_grid.m" (so in
%                 practice it will always be known in this function)
% nbr_feats     - Number of state-action features per action. Set this 
%                 value appropriately in the calling script "snake.m", to
%                 match the number of state-action features per action you
%                 end up using
%
% Output:
%
% state_action_feats - nbr_feats-by-|A| matrix, where |A| = number of
%                      possible actions (|A| = 3 in Snake), and nbr_feats
%                      is described under "Input" above. This matrix
%                      represents the state-action features extracted given
%                      the current and previous grids (game screens)
% prev_grid          - The previous grid as seen from one step in the
%                      future, i.e., prev_grid is set to the input grid
% prev_head_loc      - The previous head location as seen from one step
%                      in the future, i.e., prev_head_loc is set to the
%                      current head location (the current head location is
%                      inferred in the code below)
%
% Bugs, ideas etcetera: send them to the course email

% Extract grid size
N = size(grid, 1);

% Initialize state_action_feats to nbr_feats-by-3 matrix
state_action_feats = nan(nbr_feats, 3);

% Based on how grid looks now and at previous time step, infer head location
change_grid = grid - prev_grid;
prev_grid   = grid; % Used in later calls to "extract_state_action_features.m"

% Find head location (initially known that it is in center of grid)
if nnz(change_grid) > 0 % True, except in initial time-step
    [head_loc_m, head_loc_n] = find(change_grid > 0);
else % True only in initial time-step
    head_loc_m = round(N / 2);
    head_loc_n = round(N / 2);
end
head_loc = [head_loc_m, head_loc_n];

% Previous head location
prev_head_loc_m = prev_head_loc(1);
prev_head_loc_n = prev_head_loc(2);

% Infer current movement directory (N/E/S/W) by looking at how current and previous
% head locations are related
if prev_head_loc_m == head_loc_m + 1 && prev_head_loc_n == head_loc_n     % NORTH
    movement_dir = 1;
elseif prev_head_loc_m == head_loc_m && prev_head_loc_n == head_loc_n - 1 % EAST
    movement_dir = 2;
elseif prev_head_loc_m == head_loc_m - 1 && prev_head_loc_n == head_loc_n % SOUTH
    movement_dir = 3;
else                                                                      % WEST
    movement_dir = 4;
end

% The current head_loc will at the next time-step be prev_head_loc
prev_head_loc = head_loc;

% HERE BEGINS YOUR STATE-ACTION FEATURE ENGINEERING. ALL CODE BELOW IS 
% ALLOWED TO BE CHANGED IN ACCORDANCE WITH YOUR CHOSEN FEATURES. 
% Some skeleton code is provided to help you get started. Also, have a 
% look at the function "get_next_info" (see bottom of this function).
% You may find it useful.

[apple_loc_m, apple_loc_n] = find(grid < 0);
for action = 1 : 3 % Evaluate all the different actions (left, forward, right)
    
    % Feel free to uncomment below line of code if you find it useful
    [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc);
    
    % Replace this to fit the number of state-action features per features
    % you choose (3 are used below), and of course replace the randn() 
    % by something more sensible
    %dist = sqrt((next_head_loc(1) - apple_loc_m)^2 + (next_head_loc(2) - apple_loc_n)^2 );
    
    dist = norm([(next_head_loc(1) - apple_loc_m),(next_head_loc(2) - apple_loc_n)],1);
    
    
    if (dist == 0) 
        res = 1/2;
    else
        res = dist;
    end
    state_action_feats(1, action) =  1/res;

    
    if(grid(next_head_loc(1),next_head_loc(2)) == 1)
      state_action_feats(2, action) = 3;
    else
        future_walls = 0;
        for future_action = 1:3
            [future_head_loc, future_move_dir] = get_next_info(future_action, next_move_dir, next_head_loc);
            future_walls = future_walls + grid(future_head_loc(1),future_head_loc(2));
        end
        if(future_walls > 2)
            state_action_feats(2, action) = future_walls;
        else
            state_action_feats(2, action) = 0;
        end
        
    end
    
    grid_copy = grid;
    grid_copy(next_head_loc(1),next_head_loc(2)) = 1;
    
    %grid_copy_ones = arrayfun(@(x) replace_snake(x), grid_copy);
    grid_copy_ones = grid_copy;
   % grid_copy_ones(grid_copy_ones~=0)=1;
    
    comple_copy_grid = imcomplement(grid_copy_ones);
    comple_grid = imcomplement(grid);
    
    CC = bwconncomp(comple_grid,4);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggestSize,idx] = max(numPixels);
    
    CC_copy = bwconncomp(comple_copy_grid,4);
    numPixels_copy = cellfun(@numel,CC_copy.PixelIdxList);
    [biggestSize_copy,idx_copy] = max(numPixels_copy);
    
    if(abs(biggestSize - biggestSize_copy) > 1)
        state_action_feats(3, action) = 1;
    else
        state_action_feats(3, action) = 0;
    end
    
   
    L = labelmatrix(CC);
    %L = bwlabel(imcomplement(grid_copy_ones),4);
    %action_label = [];

    if(grid(next_head_loc(1),next_head_loc(2)) == 1)
      state_action_feats(4, action) = 0;
    else
        Label = L(next_head_loc(1),next_head_loc(2));
        if Label == idx
           state_action_feats(4, action) = 1;
        else
            state_action_feats(4, action) = 0;
        end
        %for future_action = 1:3
         %   [future_head_loc, future_move_dir] = get_next_info(future_action, next_move_dir, next_head_loc);
          %  if(grid(future_head_loc(1),future_head_loc(2)) ~= 1)
           %     action_label = [action_lable ];
            %end
         
    end 
   % end
    
    
    %state_action_feats(2, action) = randn();
    %state_action_feats(3, action) = randn();
end

end


function y = replace_snake(x)
    y = 0;
    if x >= 1
        y = 1;
    end

end 

function [next_head_loc, next_move_dir] = get_next_info(action, movement_dir, head_loc)
% Function to infer next haed location and movement direction

% Extract relevant stuff
head_loc_m = head_loc(1);
head_loc_n = head_loc(2);

if movement_dir == 1 % NORTH
    if action == 1     % left
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4; 
    elseif action == 2 % forward
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    else               % right
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    end
elseif movement_dir == 2 % EAST
    if action == 1
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    else
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    end
elseif movement_dir == 3 % SOUTH
    if action == 1
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n + 1;
        next_move_dir   = 2;
    elseif action == 2
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    else
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    end
else % WEST
    if action == 1
        next_head_loc_m = head_loc_m + 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 3;
    elseif action == 2
        next_head_loc_m = head_loc_m;
        next_head_loc_n = head_loc_n - 1;
        next_move_dir   = 4;
    else
        next_head_loc_m = head_loc_m - 1;
        next_head_loc_n = head_loc_n;
        next_move_dir   = 1;
    end
end
next_head_loc = [next_head_loc_m, next_head_loc_n];
end
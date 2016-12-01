   
    
    %grid_copy_ones = arrayfun(@(x) replace_snake(x), grid_copy);
    %grid_copy_ones = grid_copy;
    %grid_copy_ones(grid_copy_ones~=0)=1;
    grid_copy = grid;
    grid_copy(next_head_loc(1),next_head_loc(2)) = 1;
    
    comple_copy_grid = imcomplement(grid_copy);
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
         
    end 
    
    
        %--------------------------------------------------------------------
    grid_copy = grid;
    if (grid(next_head_loc(1),next_head_loc(2)) ~= 1)
       
        grid_copy(next_head_loc(1),next_head_loc(2)) = 1;

        comple_copy_grid = imcomplement(grid_copy);
        CC_copy = bwconncomp(comple_copy_grid,4);
        numPixels_copy = cellfun(@numel,CC_copy.PixelIdxList);
        [biggestSize_copy,idx_copy] = max(numPixels_copy);

        if CC_copy.NumObjects > CC.NumObjects
            state_action_feats(4, action) = 1;
        elseif (CC_copy.NumObjects < CC.NumObjects)
            state_action_feats(4, action) = 0;
        else
            state_action_feats(4, action) = 0;  
        end
    else
        state_action_feats(4, action) = 0;
    end
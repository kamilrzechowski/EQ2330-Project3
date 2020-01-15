function best_moves = bestShift(prev_frame,curr_frame, shift_directions, block_size)


    [video_height,video_width] = size(prev_frame);
    %find shift that minimise MSE
    MSE = zeros(video_height/block_size,video_width/block_size,length(shift_directions));
    best_moves = zeros(1,video_height/block_size*video_width/block_size);

    idx = 1;
    for i=1:video_height/block_size    %blocks vertically
        for j=1:video_width/block_size      %blocks horizontally
            for d=1:length(shift_directions) 
                
                %choose original area (without padding) 
                rows = 1+(i-1)*block_size : 1+(i-1)*block_size + block_size-1;
                cols = 1+(j-1)*block_size : 1+(j-1)*block_size + block_size-1;
                %shift it
                shifted_rows = rows + shift_directions(1,d);  % rows + dy
                shifted_cols = cols + shift_directions(2,d);  % rows + dx

                if(shifted_rows(1) < 1 || shifted_cols(1) < 1 || shifted_rows(block_size) > video_height || shifted_cols(block_size) > video_width)
                    MSE(i,j,d) = 99999999999999;
                    continue;
                end
                diff = (prev_frame(shifted_rows,shifted_cols)-curr_frame(rows,cols)).^2;
                MSE(i,j,d) = sum(diff(:))/numel(diff);
            end
            %find direction with minimum MSE
            best_mov = find(MSE(i,j,:) == min(MSE(i,j,:)));
            % if more than one vector has minimum MSE keep the first one
            if length(best_mov) > 1  
                best_mov = best_mov(1,1);
            end
            best_moves(1,idx) = best_mov;
            idx = idx + 1;
        end
    end
end

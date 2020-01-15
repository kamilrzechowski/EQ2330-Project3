function predicted_frame = predict_frame(current_frame,shifts, block_size, max_shift, frame_width, frame_height)

    block_count = 1;
    predicted_frame = zeros(frame_height,frame_width);
    for x = 1:block_size:frame_height-block_size+1
        for y = 1:block_size:frame_width-block_size+1

            % dy xs row(vertxcal) xndex
            % dx xs col(horxzontal) xndex
            % we are readxng blocks lxne by lxne

            dy = shifts(1,block_count);
            dx = shifts(2,block_count);
            x_top = x + dy + max_shift;
            y_top = y + dx + max_shift;
            x_bottom = x_top + block_size-1;
            y_bottom = y_top + block_size-1;
            
            predicted_frame(x:x+block_size-1,y:y+block_size-1) = ...
                current_frame(x_top:x_bottom, y_top:y_bottom);

            block_count = block_count + 1;
        end
    end
end


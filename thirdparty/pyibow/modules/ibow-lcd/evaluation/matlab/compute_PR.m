function [precision, recall] = compute_PR(loops_file, gt_file, gt_neigh, compensate, load_files)
    % Given a resulting loop and a ground truth files, this function 
    % computes the corresponding precision / recall values  
    
    % Loading files
    if load_files
        loops  = load(loops_file);
        gtruth = load(gt_file);
    else
        loops  = loops_file;
        gtruth = gt_file;
    end
    
    % Defining general counters
    TP = 0; % True Positives
    FP = 0; % False Positives
    TN = 0; % True Negatives
    FN = 0; % False Negatives
    
    loop_size = size(loops);
    gt_size = size(gtruth.truth);
    classified = zeros(1, loop_size(1));
    for i=1:loop_size(1)
        % Getting data about this loop
        % query_img = loops(i, 1);          % Current image?
        status = loops(i, 2);               % Status?
        train_id = loops(i, 3) + 1;         % What is the loop candidate?
        is_loop = status == 0;              % Is it a loop?
        gt_loop_closed = 0;                 % Is there any loop in the indicated range according to the GT file?
        gt_nloops = numel(find(gtruth.truth(i, :)));  % Number of loops in the whole GT row
        
        if is_loop
            % Selecting the range to search in the ground truth file.
            ind1 = train_id - gt_neigh;
            if ind1 < 1
                ind1 = 1;
            end
            ind2 = train_id + (gt_neigh + 1);
            if ind2 > gt_size(2)
                ind2 = gt_size(2);
            end
            gt_value = gtruth.truth(i, ind1:ind2); % Ground truth range around to the loop closure candidate
            gt_loop_closed = numel(find(gt_value)) > 0;            
        end
        
        % Taking a decision about this image
        if is_loop && gt_loop_closed
            TP = TP + 1;
            classified(1, i) = 0;
            
            % We compensate the fact that the GT has been manually labelled
            if compensate
                nprevimgs = 3;
                if i > nprevimgs
                    for j=1:nprevimgs
                        minval = train_id - 2;
                        maxval = train_id + 2;
                        if loops(i - j, 2) == 4 && loops(i - j, 3) > minval && loops(i - j, 3) < maxval
                            if classified(1, i - j) == 2
                                TN = TN - 1;
                            elseif classified(1, i - j) == 3
                                FN = FN - 1;
                            end
                        end
                    end
                end
            end
        elseif is_loop && (gt_nloops == 0 || ~gt_loop_closed)
            FP = FP + 1;
            classified(1, i) = 1;
        elseif ~is_loop && gt_nloops == 0
            TN = TN + 1;
            classified(1, i) = 2;
        elseif ~is_loop && gt_nloops > 0
            FN = FN + 1;
            classified(1, i) = 3;
        end        
    end
    
    % Printing final results.
    %disp(['TP: ', int2str(TP)]);
    %disp(['FP: ', int2str(FP)]);
    %disp(['TN: ', int2str(TN)]);
    %disp(['FN: ', int2str(FN)]);
    
    % Computing the Precision/Recall final values.
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    
    %disp(['P/R: ', num2str(precision), ' / ', num2str(recall)]);
end
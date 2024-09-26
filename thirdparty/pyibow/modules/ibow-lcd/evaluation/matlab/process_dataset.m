function process_dataset(result_dir, gt_neigh)
    % Function to process a result from a dataset
   
    % Configuring subpaths
    addpath('AcademicFigures/'); 
 
    [PR, imgvssize, imgvstime] = process(result_dir, gt_neigh, false);

    % P/R curves
    afigure;
    hold on;
    plot(PR.R, PR.P, '-o', 'MarkerIndices', length(PR.P));
    xlabel('Recall');
    ylabel('Precision');
    xlim([0.7, 1.02]);
    ylim([0.4, 1.02]);
    hold off;

    % Showing summaries
    disp('----- Summary -----');
    disp(['Max P: ', num2str(PR.P_max)]);
    disp(['Max R: ', num2str(PR.R_max)]);
    disp(['Max VWords: ', num2str(imgvssize.size(end))]);
    disp(['Avg. Time: ', num2str(mean(imgvstime.time))]);
    disp(['Std. Time: ', num2str(std(imgvstime.time))]);
end
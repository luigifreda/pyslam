% Script for obtaining the required results for IROS'18
base_dir = '/home/emilio/Escritorio/ibow-lcd/';
gt_neigh = 40;
compensate = false;

% Configuring subpaths
addpath('AcademicFigures/');

% Obtaining CityCenter results
curr_dir = strcat(base_dir, 'CityCentre/');
[PR_CC, imgvssize_CC, imgvstime_CC] = process(curr_dir, gt_neigh, compensate);
imgvstime_CC.time = smooth(imgvstime_CC.time);

curr_dir = strcat(base_dir, 'NewCollege/');
[PR_NC, imgvssize_NC, imgvstime_NC] = process(curr_dir, gt_neigh, compensate);
imgvstime_NC.time = smooth(imgvstime_NC.time);

curr_dir = strcat(base_dir, 'Lip6In/');
[PR_L6I, imgvssize_L6I, imgvstime_L6I] = process(curr_dir, gt_neigh, compensate);
imgvstime_L6I.time = smooth(imgvstime_L6I.time);

curr_dir = strcat(base_dir, 'Lip6Out/');
[PR_L6O, imgvssize_L6O, imgvstime_L6O] = process(curr_dir, gt_neigh, compensate);
imgvstime_L6O.time = smooth(imgvstime_L6O.time);

curr_dir = strcat(base_dir, 'KITTI00/');
[PR_K0, imgvssize_K0, imgvstime_K0] = process(curr_dir, gt_neigh, compensate);
imgvstime_K0.time = smooth(imgvstime_K0.time);

curr_dir = strcat(base_dir, 'KITTI05/');
[PR_K5, imgvssize_K5, imgvstime_K5] = process(curr_dir, gt_neigh, compensate);
imgvstime_K5.time = smooth(imgvstime_K5.time);

curr_dir = strcat(base_dir, 'KITTI06/');
[PR_K6, imgvssize_K6, imgvstime_K6] = process(curr_dir, gt_neigh, compensate);
imgvstime_K6.time = smooth(imgvstime_K6.time);

% P/R curves
afigure;
hold on;
plot(PR_CC.R, PR_CC.P, '-o', 'MarkerIndices', length(PR_CC.P));
plot(PR_NC.R, PR_NC.P, '-*', 'MarkerIndices', length(PR_NC.P));
plot(PR_L6I.R, PR_L6I.P, '-x', 'MarkerIndices', length(PR_L6I.P));
plot(PR_L6O.R, PR_L6O.P, '--s', 'MarkerIndices', length(PR_L6O.P));
plot(PR_K0.R, PR_K0.P, '-d', 'MarkerIndices', length(PR_K0.P));
% plot(PR_K5.R, PR_K5.P, '--^', 'MarkerIndices', length(PR_K5.P));
plot(PR_K6.R, PR_K6.P, '--p', 'MarkerIndices', length(PR_K6.P));
xlabel('Recall');
ylabel('Precision');
xlim([0.7, 1.02]);
ylim([0.4, 1.02]);
% legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'SouthWest');
legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K06', 'Location', 'SouthWest');
hold off;
print('-depsc', strcat(base_dir, 'PR_curves'));

% Images vs Size
afigure;
hold on;
%plot(imgvssize_CC.img, imgvssize_CC.size, '-o', 'MarkerIndices', length(imgvssize_CC.size));
%plot(imgvssize_NC.img, imgvssize_NC.size, '-*', 'MarkerIndices', length(imgvssize_NC.size));
%plot(imgvssize_L6I.img, imgvssize_L6I.size, '-*', 'MarkerIndices', length(imgvssize_L6I.size));
%plot(imgvssize_L6O.img, imgvssize_L6O.size, '--s', 'MarkerIndices', length(imgvssize_L6O.size));
%plot(imgvssize_K0.img, imgvssize_K0.size, '-d', 'MarkerIndices', length(imgvssize_K0.size));
%plot(imgvssize_K5.img, imgvssize_K5.size, '--^', 'MarkerIndices', length(imgvssize_K5.size));
%plot(imgvssize_K6.img, imgvssize_K6.size, '--p', 'MarkerIndices', length(imgvssize_K6.size));

plot(imgvssize_K0.img(80:end), imgvssize_K0.size(80:end), '-d', 'MarkerIndices', length(imgvssize_K0.size(80:end)));
xlabel('Time Index');
ylabel('Vocabulary Size (Words)');
xlim([80, length(imgvssize_K0.size(80:end))]);
%legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'NorthWest');
hold off;
print('-depsc', strcat(base_dir, 'imgs_vs_size'));

% Images vs Time
afigure;
hold on;
%plot(imgvstime_CC.img, imgvstime_CC.time, '-o', 'MarkerIndices', length(imgvstime_CC.time));
%plot(imgvstime_NC.img, imgvstime_NC.time, '-*', 'MarkerIndices', length(imgvstime_NC.time));
%plot(imgvstime_L6I.img, imgvstime_L6I.time, '-*', 'MarkerIndices', length(imgvstime_L6I.time));
%plot(imgvstime_L6O.img, imgvstime_L6O.time, '--s', 'MarkerIndices', length(imgvstime_L6O.time));
% plot(imgvstime_K0.img, imgvstime_K0.time, '-d', 'MarkerIndices', length(imgvstime_K0.time));
%plot(imgvstime_K5.img, imgvstime_K5.time, '--^', 'MarkerIndices', length(imgvstime_K5.time));
%plot(imgvstime_K6.img, imgvstime_K6.time, '--p', 'MarkerIndices', length(imgvstime_K6.time));

plot(imgvstime_K0.img(80:end), imgvstime_K0.time(80:end), '-d', 'MarkerIndices', length(imgvstime_K0.time(80:end)));
xlabel('Time Index');
ylabel('Avg. Time (ms)');
xlim([80, length(imgvstime_K0.time(80:end))]);
%legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'NorthWest');
hold off;
print('-depsc', strcat(base_dir, 'imgs_vs_time'));

% Showing summaries
disp('----- Summary CC -----');
disp(['Max P: ', num2str(PR_CC.P_max)]);
disp(['Max R: ', num2str(PR_CC.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_CC.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_CC.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_CC.time))]);

disp('----- Summary NC -----');
disp(['Max P: ', num2str(PR_NC.P_max)]);
disp(['Max R: ', num2str(PR_NC.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_NC.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_NC.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_NC.time))]);

disp('----- Summary L6I -----');
disp(['Max P: ', num2str(PR_L6I.P_max)]);
disp(['Max R: ', num2str(PR_L6I.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_L6I.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_L6I.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_L6I.time))]);

disp('----- Summary L6O -----');
disp(['Max P: ', num2str(PR_L6O.P_max)]);
disp(['Max R: ', num2str(PR_L6O.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_L6O.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_L6O.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_L6O.time))]);

disp('----- Summary KITTI 00 -----');
disp(['Max P: ', num2str(PR_K0.P_max)]);
disp(['Max R: ', num2str(PR_K0.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K0.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K0.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K0.time))]);

disp('----- Summary KITTI 05 -----');
disp(['Max P: ', num2str(PR_K5.P_max)]);
disp(['Max R: ', num2str(PR_K5.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K5.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K5.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K5.time))]);

disp('----- Summary KITTI 06 -----');
disp(['Max P: ', num2str(PR_K6.P_max)]);
disp(['Max R: ', num2str(PR_K6.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K6.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K6.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K6.time))]);

close all;
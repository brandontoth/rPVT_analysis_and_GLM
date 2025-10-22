% set behaviors that you want to combine across
behaviors = {'climbing', 'rearing', 'scratching', 'walking'};
% string that we're looking for in filenames
target    = 'Mouse_velocity.xlsx';

% get desired folders in the current directory
D = dir;
% D = D(ismember({D.name}, behaviors));
D = D(~ismember({D.name}, {'.', '..'}));
D = D([D(:).isdir]);

% loop through folders and get velocity data
for k = 1:numel(D)
    currD = D(k).name; % get the current subdirectory name
    cd(currD)          % change the directory

    fprintf(1, 'Now reading %s\n', currD);

    velo_file = dir;
    velo_file = velo_file(ismember({velo_file.name}, {target}));
    velo = readtable(velo_file.name);

    all_velo(:, k) = velo{:, 2};

    clear velo_file velo

    cd('..')
end

% remove nans and combine across columns
all_velo(isnan(all_velo)) = 0;
all_velo = sum(all_velo, 2);

% convolve with gaussian window to smooth data
win = gausswin(round(2.3 * 50));  % std of about 1 sec
win = win ./ sum(win);
conv_velo = conv(all_velo, win, 'same');

% sanity check
figure; plot(conv_velo); xlim([0 length(conv_velo)])

%%
cam_fs=vid.NumFrames/(length(z_fp_sig)/Fs);
% light_interp = interp1(times, my_light, t_target, 'pchip', 'extrap');
t_original = (0:numel(conv_velo)-1) / cam_fs;
% Target time vector (1000 Hz)
t_target = 0:1/1000:t_original(end);
% Interpolate using shape-preserving cubic interpolation
interp_velo = interp1(t_original, conv_velo, t_target, 'pchip');

%%
[corr, lag] = xcorr(zscore(stretch_velo), z_fp_sig, 100 * Fs, "normalized");
figure;plot(lag / Fs, corr);
box off
pbaspect([1 1 1])
ax = gca; ax.FontSize = 12; 
xlabel("Time (s)", "FontSize", 12)
ylabel("Norm. correlation", "FontSize", 12)
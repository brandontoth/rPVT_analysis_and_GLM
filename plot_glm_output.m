function [] = plot_glm_output(triggers, cutTime, photoFolder, fileName, saveFigures, Title, Fs, color, handle)

switch handle
    case 'plot'
        figure
    case 'hold'
        hold on
end

set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 0.35 0.55])
sgtitle(Title)

a2 = subplot(4, 5, 2);
shadedErrorBar([], mean(triggers.triggerOmittedLight), ...
    std(triggers.triggerOmittedLight) / sqrt(size(triggers.triggerOmittedLight, 1)), ...
    'lineProps', color)
xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
xt = get(gca, 'xtick');
set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
xlim([0 length(triggers.triggerOmittedLight)])
ylim([-1 2])
ylabel('Omission')
title('Light', 'FontWeight', 'normal')
pbaspect([1 1 1])

if isfield(triggers, 'triggerPrematurePoke')
    a3 = subplot(4, 5, 6);
    shadedErrorBar([], mean(triggers.triggerTone), ...
    std(triggers.triggerTone) / sqrt(size(triggers.triggerTone, 1)), 'lineProps', color)
    xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
    xt = get(gca, 'xtick');
    set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
    xlim([0 length(triggers.triggerTone)])
    ylim([-1 2])
    title('Tone', 'FontWeight', 'normal')
    pbaspect([1 1 1])
        
    a4 = subplot(4, 5, 8);
    
    if size(triggers.triggerPrematurePoke, 1) == 1
        stdErr = zeros(1, length(triggers.triggerPrematurePoke));
        shadedErrorBar(1:length(triggers.triggerPrematurePoke), ...
            triggers.triggerPrematurePoke, stdErr, 'lineProps', color);
    else
        stdErr = std(triggers.triggerPrematurePoke) / sqrt(size(triggers.triggerPrematurePoke, 1));
        shadedErrorBar(1:length(triggers.triggerPrematurePoke), ...
            mean(triggers.triggerPrematurePoke), stdErr, 'lineProps', color);
    end

    xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
    xt = get(gca, 'xtick');
    set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
    xlim([0 length(triggers.triggerOmittedLight)])
    ylim([-1 2])
    ylabel('Premature')
    title('Poke', 'FontWeight', 'normal')
    pbaspect([1 1 1])
else
    a3 = subplot(4, 5, 6); ylabel('Premature')
    a4 = subplot(4, 5, 8);
end

if isfield(triggers, 'triggerIncorrectPoke')
   
    a6 = subplot(4, 5, 12);
    
    if size(triggers.triggerIncorrectLight, 1) == 1
        stdErr = zeros(1, length(triggers.triggerIncorrectLight));
        shadedErrorBar(1:length(triggers.triggerIncorrectLight), ...
            triggers.triggerIncorrectLight, stdErr, 'lineProps', color);
    else
        stdErr = std(triggers.triggerIncorrectLight) / sqrt(size(triggers.triggerIncorrectLight, 1));
        shadedErrorBar(1:length(triggers.triggerIncorrectLight), ...
            mean(triggers.triggerIncorrectLight), stdErr, 'lineProps', color);
    end
    
    xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
    xt = get(gca, 'xtick');
    set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
    xlim([0 length(triggers.triggerOmittedLight)])
    ylabel('Incorrect')
    ylim([-1 2])
    pbaspect([1 1 1])
    
    a7 = subplot(4, 5, 13);
    
    if size(triggers.triggerIncorrectPoke, 1) == 1
        stdErr = zeros(1, length(triggers.triggerIncorrectPoke));
        shadedErrorBar(1:length(triggers.triggerIncorrectPoke), ...
            triggers.triggerIncorrectPoke, stdErr, 'lineProps', color);
    else
        stdErr = std(triggers.triggerIncorrectPoke) / sqrt(size(triggers.triggerIncorrectPoke, 1));
        shadedErrorBar(1:length(triggers.triggerIncorrectPoke), ...
            mean(triggers.triggerIncorrectPoke), stdErr, 'lineProps', color);
    end
    
    xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
    xt = get(gca, 'xtick');
    set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
    xlim([0 length(triggers.triggerOmittedLight)])
    ylim([-1 2])
    pbaspect([1 1 1])
else
    a6 = subplot(4, 5, 12);
    a7 = subplot(4, 5, 13);
end

a9 = subplot(4, 5, 17);
shadedErrorBar([], mean(triggers.triggerCorrectLight), ...
    std(triggers.triggerCorrectLight) / sqrt(size(triggers.triggerCorrectLight, 1)), 'lineProps', color)
xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
xt = get(gca, 'xtick');
set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
xlim([0 length(triggers.triggerOmittedLight)])
ylabel('Correct')
ylim([-1 2])
pbaspect([1 1 1])

a10 = subplot(4, 5, 18);
shadedErrorBar([], mean(triggers.triggerCorrectPoke), ...
    std(triggers.triggerCorrectPoke) / sqrt(size(triggers.triggerCorrectPoke, 1)), 'lineProps', color)
xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
xt = get(gca, 'xtick');
set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
xlim([0 length(triggers.triggerOmittedLight)])
ylim([-1 2])
pbaspect([1 1 1])

a12 = subplot(4, 5, 20);
shadedErrorBar([], mean(triggers.triggerPellet), ...
    std(triggers.triggerPellet) / sqrt(size(triggers.triggerPellet, 1)), 'lineProps', color)
xline(cutTime, 'Color',[1 1 1] * 0.7,'LineStyle',':', 'LineWidth', 1);
xt = get(gca, 'xtick');
set(gca, 'XTick', xt, 'xticklabel', (xt - cutTime) / Fs)
xlim([0 length(triggers.triggerOmittedLight)])
ylim([-1 2])
title('Pellet retrieval', 'FontWeight', 'normal')
pbaspect([1 1 1])

linkaxes([a2 a3 a4 a6 a7 a9 a10 a12], 'x')
set(gcf,'renderer','Painters')

if saveFigures == 1
    print((horzcat(photoFolder, '/', fileName)), '-dpdf')
end

end
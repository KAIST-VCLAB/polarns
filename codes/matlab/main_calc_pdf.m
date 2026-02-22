%% Figure 3 noise model



%% check pdf vaildity

angles = -1:0.01:1;
angles = angles.*pi;

conf_interval_95 = [];
conf_interval_99 = [];
vars = [];
str_for_legend = [];
figure('Position', [1000 1000 1200 300]);
subplot(1,3,3);
box on;
hold on;
first_plot = 1;
old_solution_95 = -pi;
old_solution_99 = -pi;
for i = -1:1
    ratio_polar_noise = 2.^i;
    fun = @(x) pdf_aolp(x,ratio_polar_noise);
    integration = integral(fun,-pi,pi);
    if abs(integration-1) > 0.01
        fprintf("wrong pdf: %f\n",integration);
    end
    x0 = 0;
    options = optimoptions('fmincon','Display','off');
    x = fmincon(@(x) obj_confidence_interval_loss(x,fun,0.05./2) ,x0,[],[],[],[],old_solution_95,pi,[],options);
    conf_interval_95 = [conf_interval_95;x];
    old_solution_95 = x;
    x = fmincon(@(x) obj_confidence_interval_loss(x,fun,0.01./2) ,x0,[],[],[],[],old_solution_99,pi,[],options);
    conf_interval_99 = [conf_interval_99;x];
    old_solution_99 = x;
    
    probs = fun(angles);

    vars = [vars var(angles,probs./sum(probs))];

    plot(rad2deg(angles)/2, probs*2*pi/180,'LineWidth',0.5);
%     polarplot(angles, probs,'LineWidth',0.5.5);
    if first_plot
        % axis([-90,90,0,inf]);
        axis([-90,90,0,0.03]);
        hold on;
        first_plot=0;
    end
    str_for_legend = [str_for_legend;"s_{pol}/\sigma="+sprintf("%d",i)];
end
% legend('-2','-1','0','1','2');


% str_for_legend = ["s_{pol}/\sigma_{v}=0.25",
%     "s_{pol}/\sigma_{v}=0.5",
%     "s_{pol}/\sigma_{v}=1",
%     "s_{pol}/\sigma_{v}=2",
%     "s_{pol}/\sigma_{v}=4"];
% str_for_legend = ["s_{pol}/\sigma_{v}=0.5",
%     "s_{pol}/\sigma_{v}=1",
%     "s_{pol}/\sigma_{v}=2"];
str_for_legend = ["0.5",
    "1",
    "2"];

legend(str_for_legend);
xlabel('Angle difference (deg)');
% title('PDF of AoLP noise');
ylabel('Probability density');
xticks(-90:45:90);
yticks(0:0.01:0.03);
fontname('Times New Roman')
fontsize(16,"points")
hold off;
conf_interval_95_deg = rad2deg(conf_interval_95);
stddevs = sqrt(vars);
%% ~ 10 deg
% fun = @(x) pdf_aolp(x,10);
% x = fmincon(@(x) obj_confidence_interval_loss(x,fun,0.05./2) ,x0,[],[],[],[],-0.5,pi,[],options)
% pd = makedist('Normal','sigma',sqrt(10));
% ci = paramci(pd)


%% DoLP by dolp

est_dolp = 0:0.01:1;

str_for_legend = [];
subplot(1,3,1);
box on;
hold on;
first_plot = 1;

true_dolp = [0.1,0.2,0.4];
ratio_s0_noise = 10;

max_idx = zeros(3,1);

for i = 1:3
    fun = @(x) pdf_dolp(x,true_dolp(i),1.0./ratio_s0_noise);
    
    probs = fun(est_dolp);

    [~,max_idx(i)] = max(probs);

    plot(est_dolp, probs,'LineWidth',0.5);
%     polarplot(angles, probs,'LineWidth',0.5.5);
    if first_plot
        % axis([-90,90,0,inf]);
        axis([0,0.8,0,6]);
        hold on;
        first_plot=0;
    end
    str_for_legend = [str_for_legend;"s_{pol}/\sigma="+sprintf("%d",i)];
end
% legend('-2','-1','0','1','2');


% str_for_legend = ["s_{pol}/\sigma_{v}=0.25",
%     "s_{pol}/\sigma_{v}=0.5",
%     "s_{pol}/\sigma_{v}=1",
%     "s_{pol}/\sigma_{v}=2",
%     "s_{pol}/\sigma_{v}=4"];
str_for_legend = ["DoLP=0.1",
    "DoLP=0.2",
    "DoLP=0.4"];

legend(str_for_legend);
xlabel('Measured DoLP');
% title('PDF of AoLP noise');
ylabel('Probability density');
% xticks(-90:45:90);
% yticks(0:0.01:0.03);

xline(0.1,'LineWidth',0.5,'HandleVisibility','off','Color',"#0072BD"); % blue
xline(0.2,'LineWidth',0.5,'HandleVisibility','off','Color',"#D95319"); % red
xline(0.4,'LineWidth',0.5,'HandleVisibility','off','Color',"#EDB120"); % orange

xline(est_dolp(max_idx(1)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#0072BD"); % blue
xline(est_dolp(max_idx(2)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#D95319"); % red
xline(est_dolp(max_idx(3)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#EDB120"); % orange
fontname('Times New Roman')
fontsize(16,"points")
hold off;


%% DoLP

est_dolp = 0:0.01:1;

vars = [];
str_for_legend = [];
subplot(1,3,2);
box on;
hold on;
first_plot = 1;
old_solution_95 = -pi;
old_solution_99 = -pi;

true_dolp = 0.2;
ratio_s0_noise = [5 10 20];
max_idx = zeros(3,1);


for i = 1:3
    fun = @(x) pdf_dolp(x,true_dolp,1.0./ratio_s0_noise(i));
    
    probs = fun(est_dolp);

    [~,max_idx(i)] = max(probs);

    plot(est_dolp, probs,'LineWidth',0.5);
%     polarplot(angles, probs,'LineWidth',0.5.5);
    if first_plot
        % axis([-90,90,0,inf]);
        axis([0,0.8,0,10]);
        hold on;
        first_plot=0;
    end
    str_for_legend = [str_for_legend;"s_{pol}/\sigma="+sprintf("%d",i)];
end
% legend('-2','-1','0','1','2');


% str_for_legend = ["s_{pol}/\sigma_{v}=0.25",
%     "s_{pol}/\sigma_{v}=0.5",
%     "s_{pol}/\sigma_{v}=1",
%     "s_{pol}/\sigma_{v}=2",
%     "s_{pol}/\sigma_{v}=4"];
% str_for_legend = ["s_{0}/\sigma_{v}=5",
%     "s_{0}/\sigma_{v}=10",
%     "s_{0}/\sigma_{v}=20"];
str_for_legend = ["5",
    "10",
    "20"];

legend(str_for_legend);
xlabel('Measured DoLP');
% title('PDF of AoLP noise');
ylabel('Probability density');
% xticks(-90:45:90);
% yticks(0:0.01:0.03);

xline(0.2,'LineWidth',0.5,'HandleVisibility','off');

xline(est_dolp(max_idx(1)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#0072BD"); % blue
xline(est_dolp(max_idx(2)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#D95319"); % red
xline(est_dolp(max_idx(3)),'--','LineWidth',0.5,'HandleVisibility','off','Color',"#EDB120"); % orange
fontname('Times New Roman')
fontsize(16,"points")
hold off;

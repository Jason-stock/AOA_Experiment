function [] = result_plot_graph(X, Y_target, Y_predict)
    plot(X,Y_target, 'b-');
    hold on;
    plot(X,Y_predict,'r-');
    hold off;
    legend('target function','proposed approach','Location', 'northeastoutside');
end


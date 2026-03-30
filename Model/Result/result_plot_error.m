function [] = result_plot_error(X_train, error_train, X_test, error_test)
    plot(X_train, error_train, 'b-');
    hold on;
    plot(X_test, error_test,'g-');
    hold off;
    legend('train error', 'test error', 'Location', 'northeastoutside');
end


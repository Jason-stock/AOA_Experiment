function [] = result_plot_lrnCurve(tIter, error)
    plot(1:tIter, error, "ro-");
end
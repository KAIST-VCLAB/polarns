function loss = obj_confidence_interval_loss(x,fun,left_conf_interval)
integration = integral(fun,-pi,x);
loss = 1000000*(left_conf_interval-integration).^2;
end


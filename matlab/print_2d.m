function [str] = print_2d(m)
    str = "[\n";
    for i = 1: size(m, 1)
        str =  strcat(str, "\t[");
        for j = 1: size(m,2)
            if j < size(m,2)
                str = strcat(str, string(m(i, j)), ", ");
            else
                str = strcat(str, string(m(i, j)), "]");
            end
        end
        if i < size(m, 1)
            str = strcat(str, ",\n");
        else
            str = strcat(str, "\n]");
        end
    end
end
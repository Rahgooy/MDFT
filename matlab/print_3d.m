function [str] = print_3d(m)
    str = "[\n";
    for k = 1: size(m, 3)
        str =  strcat(str, "  [\n");
        for i = 1: size(m, 1)
            str =  strcat(str, "\t[");
            for j = 1: size(m,2)
                if j < size(m,2)
                    str = strcat(str, string(m(i, j, k)), ", ");
                else
                    str = strcat(str, string(m(i, j, k)), "]");
                end
            end
            if i < size(m, 1)
                str = strcat(str, ",\n");
            else
                str = strcat(str, "\n  ]");
            end
        end
        if k < size(m, 3)
            str = strcat(str, ",\n");
        else
            str = strcat(str, "\n]\n");
        end
    end
end
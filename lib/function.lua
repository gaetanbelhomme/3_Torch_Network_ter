--function split(s, delimiter)
--    result = {};
--    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
--        table.insert(result, match);
--    end
--    return result;
--end
function split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={} ; i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

"""
    save_geometry_info(cdir::String, geometry::Geometry, msg::String)

Save information about geometry.
"""
function save_geometry_info(cdir::String, geometry::Geometry, msg::String)
    fpath = joinpath(cdir, "Description.txt")
    open(fpath, "w") do f
        println(f, msg)
        println(f, "Source:")
        println(f, geometry.source)
        println(f, "Observer:")
        println(f, geometry.observer)
        println(f, "BH spin")
        println(f, "a = $(geometry.a)")
        println(f, "ODE Options")
        println(f, geometry.ode_options)
        println(f, geometry.opt_options)
    end
end


"""
    save_config_info(config::Dict{Symbol, Any})

Save information about the configuration file.
"""
function save_config_info(config::Dict{Symbol, Any})
    fpath = joinpath(checkpointdir(config), "Description.txt")
    open(fpath, "w") do f
        for (key, value) in config
            println(f, "$key:")
            println(f, value)
            println(f, "\n")
        end
    end
end

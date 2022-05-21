"""
    save_geometry_info(cdir::String, runID::Int64, geometry::Geometry, msg::String)

Save information about geometry.
"""
function save_geometry_info(cdir::String, runID::String, geometry::Geometry, msg::String)
    fpath = joinpath(cdir, "$(runID)_Description.txt")
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

using HDF5
using JLD2
using FilePathsBase

# Specify your existing JLD2 file path
input_file_path = "../data/trajectories/random_trajectories_2023_10_14_19_22_04.jld2"

# Load your existing JLD2 data
@load input_file_path results

# Derive HDF5 file path from JLD2 file path
output_file_path = replace(input_file_path, ".jld2" => ".h5")

# Create the HDF5 file
h5open(output_file_path, "w") do file
    # Create groups or datasets as needed
    for (i, traj) in enumerate(results)
        grp = create_group(file, "Trajectory_$i")
        for (key, value) in traj
            write(grp, string(key), value)
        end
    end
end


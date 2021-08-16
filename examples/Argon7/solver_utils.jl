using LinearAlgebra: dot
using Plots
function init_position(c::Potentials.Configuration, dim)
    #=
    Initialise the positions for a series 
    of particles.
    
    Parameters
    ----------
    N: int
        Number of particles in the 
        system
    dim: int
        Number of dimensions
    
    Returns
    -------
    ndarray of floats
        Initial velocities for a series of 
        particles (eVs/Åamu)
    =#
    N = c.num_atoms
    if dim == 2
        N = N-1
        r0 = 2 .*[cos.(2*pi/N .* (1:N))+1e-2*randn(N) sin.(2*pi/N.*(1:N))+1e-2*randn(N) 0.0 .*(1:N)]
        r0 = [r0; [0 0 0]]
    elseif dim == 3
        if N <= 7
            r0 = [-0.9523699364        0.0159052548       -0.0840802250;
            -0.2528949024        0.8598484003       -0.3332183372;
            -0.3357005052       -0.8500181306        0.2812584200;
            0.7960710440        0.5155096551       -0.1218613860;
            0.7448954577       -0.5412454414        0.2579017563;
            -0.0442130155        0.1953980103        0.5377653956;
            0.0442118576       -0.1953977484       -0.5377656238]
        elseif N == 38
            r0 = [0.163946667699999993 0.319538517000000022 1.71226674359999986;
            1.18570812759999988 -1.12890812809999996 -0.616849616600000039;
            1.47183871269999988 -0.0530362826999999995 0.944308599800000037;
            -0.156601128000000006 1.44806844500000009 -0.969234048100000023;
            1.43644861639999988 0.381943172100000006 -0.92280851779999995;
            -0.659303829199999969 -1.58099572070000005 -0.355825472500000017;
            -1.43644861639999988 -0.381943172100000006 0.92280851779999995;
            0.659303829199999969 1.58099572070000005 0.355825472600000026;
            -1.18570812749999988 1.12890812809999996 0.616849616600000039;
            0.156601128099999987 -1.44806844500000009 0.969234048100000023;
            -1.47183871259999988 0.0530362826999999995 -0.944308599700000029;
            -0.692522445699999989 0.622174532900000021 -1.48126712230000002;
            -0.903125893300000038 0.439946115799999982 1.43235882759999988;
            -0.943262934499999983 -0.888676767300000003 -1.17530822099999988;
            -0.163946667699999993 -0.319538517000000022 -1.71226674359999986;
            0.402594671899999978 -1.70081948270000005 -0.0772747840999999935;
            -1.71903085060000005 0.307018840199999976 0.107299306900000002;
            0.905297373100000025 1.32824468299999987 -0.690683359700000055;
            -0.402594671899999978 1.70081948270000005 0.0772747840999999935;
            1.71903085060000005 -0.307018840199999976 -0.107299306799999994;
            -0.905297373000000016 -1.32824468299999987 0.690683359700000055;
            0.943262934599999991 0.888676767300000003 1.17530822109999988;
            0.692522445799999997 -0.622174532900000021 1.48126712230000002;
            0.903125893400000046 -0.439946115799999982 -1.43235882749999988;
            -1.18439743100000006 -0.634064735699999993 -0.125267553799999987;
            0.120331581600000001 -1.00573849309999996 -0.891368452299999947;
            0.934263338099999952 -0.873132691499999947 0.430486517999999985;
            1.18439743110000006 0.634064735699999993 0.125267553799999987;
            -0.934263338000000054 0.873132691499999947 -0.430486517899999976;
            -0.370465674600000017 -0.501458934099999976 1.19658741639999988;
            0.370465674700000025 0.501458934099999976 -1.19658741639999988;
            -0.120331581600000001 1.00573849309999996 0.891368452299999947;
            0.649793600800000037 -0.185104513500000012 -0.381540884400000002;
            -0.405362064899999985 -0.0660416075000000019 -0.658322833699999999;
            0.124574169400000004 0.750628853799999995 -0.15200806280000001;
            -0.649793600800000037 0.185104513500000012 0.381540884400000002;
            0.405362064999999994 0.0660416075000000019 0.658322833699999999;
            -0.124574169299999996 -0.750628853799999995 0.15200806280000001]
        else
            r0 = rand(N, d)
        end

        r0 += 1e-2*randn(size(r0))
        com = mean(r0, dims = 1) 
        r0 = r0 .- com
    else
        return false
    end

    for j = 1:c.num_atoms
        c.Positions[j] = Potentials.Position( r0[j, 1], r0[j, 2], r0[j, 3], c.Positions[j].type)
    end
    
    return c
end

function init_velocity(c::Potentials.Configuration, dim::Int, T::Float64)
    #=
    Initialise the velocities for a series 
    of particles.
    
    Parameters
    ----------
    T: float
        Temperature of the system at 
        initialisation (K)
    number_of_particles: int
        Number of particles in the 
        system
    
    Returns
    -------
    ndarray of floats
        Initial velocities for a series of 
        particles (eVs/Åamu)
    =#
    N = c.num_atoms
    r0 = randn(N, 3) .- 0.5
    if dim == 2
        r0[:, 3] .= 0.0
    end
    r0 = r0 * T
    for j = 1:c.num_atoms
        c.Velocities[j] = Potentials.Position( r0[j, 1], r0[j, 2], r0[j, 3], c.Positions[j].type)
    end 
    return c
end



function update_velo(v::Vector, a::Vector, a1::Vector, dt::Float64)
    #= 
    Update the particle velocities.
    
    Parameters
    ----------
    v: ndarray of floats
        The velocities of the particles in a 
        single dimension (eVs/Åamu)
    a: ndarray of floats
        The accelerations of the particles in a 
        single dimension at the previous 
        timestep (eV/Åamu)
    a1: ndarray of floats
        The accelerations of the particles in a 
        single dimension at the current 
        timestep (eV/Åamu)
    dt: float
        The timestep length (s)
    
    Returns
    -------
    ndarray of floats:
        New velocities of the particles in a 
        single dimension (eVs/Åamu)
    =#
    return v + 0.5 * (a + a1) * dt
end

function verlet(p::Potentials.ArbitraryPotential, c::Potentials.Configuration, dt::Float64, Temp:: Float64; γ = 1.0, k_b = 1.0/119.8)
   #= 
    Update the particle positions and velocities by solving.
    dx/dt = dv/dt
    dv/dt = F(r)/m - γ * v + R/m = a
    
    Parameters
    ----------
    x : ndarray of floats
        The positions of the particles (Å)
    v: ndarray of floats
        The velocities of the particles
        single dimension (eVs/Åamu)
    dt: float
        The timestep length (s)
    
    Returns
    -------
    ndarray of floats:
        New velocities and positions of the particles in a 
        single dimension (Å, eVs/Åamu)
    =#
    pos = vec(c.Positions)
    vels = vec(c.Velocities)
    R = sqrt(2*Temp*k_b*dt*γ)
    a = Potentials.force(c, p)    
    a = a - γ .* vels
    noise = 0.5*R*randn(c.num_atoms, 3)
    if pos[1][3] == 0.0
        for j = 1:length(a)
            a[j][3] = 0.0
            noise[j, 3] = 0.0
        end
    end
    v_star = update_velo(vels, a, 0.0*a, dt)
    noise = noise .- mean( noise, dims = 1 )
    noise = [[noise[i, j] for j = 1:3] for i = 1:c.num_atoms]
    v_star += 0.5*noise
    # Second half-time step
    x_new = c.Positions + v_star*dt
    # println("Old Positions ")
    # show(stdout, "text/plain", c.Positions)
    # println(" ")
    # println("New positions ")
    # show(stdout, "text/plain", x_new)
    # println(" ")
    for (i, xi) in enumerate(x_new)       
        c.Positions[i] = xi
    end
    a_new = Potentials.force(c, p)
    a_new = a_new - γ .* vels
    noise = 0.5*R*randn(c.num_atoms, 3)
    if pos[1][3] == 0.0
        for j = 1:length(a)
            a_new[j][3] = 0.0
            noise[j, 3] = 0.0
        end
    end
    noise = noise .- mean( noise, dims = 1 )
    noise = [[noise[i, j] for j = 1:3] for i = 1:c.num_atoms]
    v_new = update_velo(v_star, a, a_new, dt)
    v_new += 0.5*noise

    # Save
    for (i, (xi, vi)) in enumerate(zip(x_new, v_new))
        c.Positions[i] = xi
        c.Velocities[i] = Potentials.Position(vi[1], vi[2], vi[3], c.Positions[i].type)
    end
    return c
end

function solve(p::Potentials.ArbitraryPotential, c0::Potentials.Configuration, dt::Float64, Nt::Int, Temp ::Array{Float64}; save_dt = 1 :: Int)
    N = Int(Nt / save_dt)
    r = Vector{Potentials.Configuration}(undef, N)
    r = fill!(r, c0)

    t = zeros( (N, 1)); 
    
    τ = 0.0
    count = 1
    c_temp = deepcopy(c0)
    for j = 2:(Nt)
        τ +=  dt
        # println("Time ", τ)
        c_temp = verlet(p, c_temp, dt, Temp[j-1])
        if (j-1) % save_dt == 0
            count += 1
            r[count] = deepcopy(c_temp)
            t[count] = τ
        end
    end
    
    return r, t
end

function get_positions(r::Vector{PotentialUQ.Potentials.Configuration})
    N = length(r)
    num_atoms = r[1].num_atoms

    pos = Vector{Vector{Float64}}(undef, N)
    for (i, ri) in enumerate(r)
        pos[i] = PotentialUQ.Potentials.norm.(ri.Positions)
    end
    return pos 
end

function get_velocities(r::Vector{Potentials.Configuration})
    N = length(r)
    num_atoms = r[1].num_atoms

    pos = Vector{Vector{Float64}}(undef, N)
    for (i, ri) in enumerate(r)
        pos[i] = Potentials.norm.(ri.Velocities)
    end
    return pos 
end

function calculate_rdf(r::Vector{Potentials.Configuration}; L = 2.5, maxbin = 1000)

    bins = vcat(range(0, L; length = maxbin+1)...)
    rdf = zeros(maxbin)
    bin_centers = 0.5 .* (bins[1:end-1] + bins[2:end])
    n = length(r)
    num_atoms = r[1].num_atoms
    dr = bin_centers[2] - bin_centers[1]
    hist = zeros(maxbin)
    for c ∈ r
        distances = Potentials.get_interparticle_distance(c)
        for d in distances 
            bin = ceil(Int, d / dr)
            if (bin > 1) && (bin < maxbin)
                hist[bin] += 1
            end
        end
    end
    
    volume = 4 / 3 * π * L^3
    density = num_atoms / volume
    rdf = zeros(maxbin)
    for bin = 1:maxbin
        rlower = bins[bin]
        rupper = bins[bin+1]
        volume_of_shell = 4/3 * π * (rupper^3 - rlower^3)
        rdf[bin] = (hist[bin] / n) / volume_of_shell / density
    end

    return (bin_centers, rdf)
end

function calculate_mean_squared_displacement(r::Vector{Potentials.Configuration})
    n = length(r)
    num_atoms = r[1].num_atoms
    p0 = vec(r[1].Positions)
    disp = zeros(n)
    for i = 1:n
        for j = 1:num_atoms
            p = vec(r[i].Positions)
            dr = p - p0
            disp[i] += dot(dr, dr) / num_atoms
        end
    end
    return disp
end

function calculate_mass_diffusivity(r::Vector{Potentials.Configuration}, t)
    disp = calculate_mean_squared_displacement(r)
    D = sum( disp[end-100:end] ./ (6 .* t[end-100:end] ) ) / 100
    return D
end

function animate_atoms(r::Vector{Potentials.Configuration}, t; dim = 2, fps = 60, dT = 20, title = "lennard_jones_cluster.gif")
    n = length(r)
    num_atoms = r[1].num_atoms
    
    if dim == 3
        
        anim = @animate for i = 1:dT:n
            pos = hcat(vec(r[i].Positions)...)
            scatter(pos[1, :], pos[2, :], pos[3, :], xaxis = ("x", (-2, 2), -2:0.5:2), 
                            yaxis = ("y", (-2, 2), -2:0.5:2),
                            zaxis = ("z", (-2, 2), -2:0.5:2),
                            markersize = 20, 
                            c=colormap("Blues",num_atoms), 
                            legend = false,
                            title = string("τ = ", floor(t[i]) ))
        end
    
    elseif dim==2
        anim = @animate for i = 1:dT:n
            pos = hcat(vec(r[i].Positions)...)
            scatter(pos[1, :], pos[2, :], xaxis = ("x", (-4, 4), -4:0.5:4), 
                            yaxis = ("y", (-2, 2), -4:0.5:4),
                            c=colormap("Blues",num_atoms),
                            legend = false, 
                            title = string("τ = ", floor(t[i]) ) )
            
        end
    end
    
    gif(anim, title, fps = fps)
end


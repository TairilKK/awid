using Random
using Flux

include("base.jl")
include("losses.jl")
include("graphNodes.jl")
include("operators.jl")
include("operatorPasses.jl")
include("operatorDiscretization.jl")

function compare_dense_my_vs_flux(; in_dim=5, out_dim=3, seed=1234, T=Float64)
  Random.seed!(seed)

  xval = randn(T, in_dim)
  Wval = randn(T, out_dim, in_dim)
  bval = randn(T, out_dim)

  # ===== Mój framework =====
  input = tensor(in_dim)

  dense_node = dense(in_dim => out_dim)(input)
  mul_node, b_node = dense_node.args
  W_node, _ = mul_node.args

  W_node.data .= Wval
  b_node.data .= bval

  model_my = graph(dense_node)

  zerograd!(model_my)
  forward!(model_my, input => xval)
  backward!(model_my)   # seed.grad .= 1 => odpowiada loss = sum(output)

  y_my = copy(dense_node.data)
  gx_my = copy(input.grad)
  gW_my = copy(W_node.grad)
  gb_my = copy(b_node.grad)

  # ===== Flux =====
  layer = Flux.Dense(in_dim => out_dim)
  layer.weight .= Wval
  layer.bias .= bval

  loss_flux(x) = sum(layer(x))

  y_flux = layer(xval)

  flux_grads = Flux.gradient(Flux.params(layer)) do
    loss_flux(xval)
  end

  gx_flux = Flux.gradient(loss_flux, xval)[1]
  gW_flux = flux_grads[layer.weight]
  gb_flux = flux_grads[layer.bias]

  println("=== OUTPUT ===")
  println("max |y_my - y_flux|    = ", maximum(abs.(y_my .- y_flux)))

  println("\n=== INPUT GRAD ===")
  println("max |dx_my - dx_flux| = ", maximum(abs.(gx_my .- gx_flux)))

  println("\n=== WEIGHT GRAD ===")
  println("max |dW_my - dW_flux| = ", maximum(abs.(gW_my .- gW_flux)))

  println("\n=== BIAS GRAD ===")
  println("max |db_my - db_flux| = ", maximum(abs.(gb_my .- gb_flux)))

  println("\n=== FULL VALUES ===")
  println("y_my = ", y_my)
  println("y_flux = ", y_flux)

  println("\ngx_my = ", gx_my)
  println("gx_flux = ", gx_flux)

  println("\ngW_my = ")
  display(gW_my)
  println("gW_flux = ")
  display(gW_flux)

  println("gb_my = ", gb_my)
  println("gb_flux = ", gb_flux)

  return (
    y_my=y_my,
    y_flux=y_flux,
    gx_my=gx_my,
    gx_flux=gx_flux,
    gW_my=gW_my,
    gW_flux=gW_flux,
    gb_my=gb_my,
    gb_flux=gb_flux,
  )
end

function compare_maxpool_my_vs_flux(; C=1, H=4, W=4, kernel=(2, 2), seed=1234, T=Float64)
  Random.seed!(seed)

  kh, kw = kernel
  @assert H % kh == 0
  @assert W % kw == 0

  # wejście bez remisów w oknach poolingowych
  xval = randn(T, C, H, W)
  xval .+= reshape(T.(1:length(xval)) .* T(1e-8), size(xval))

  # ===== Mój framework =====
  input = tensor(C, H, W)
  pool_node = maxpool(kernel)(input)
  model_my = graph(pool_node)

  zerograd!(model_my)
  forward!(model_my, input => xval)
  backward!(model_my)   # seed.grad .= 1 => loss = sum(output)

  y_my = copy(pool_node.data)
  gx_my = copy(input.grad)

  # ===== Flux =====
  # moje: (C,H,W)
  # Flux: (H,W,C,N)
  x_flux = permutedims(xval, (2, 3, 1))
  x_flux = reshape(x_flux, H, W, C, 1)

  layer = Flux.MaxPool(kernel)

  loss_flux(x) = sum(layer(x))

  y_flux_raw = layer(x_flux)
  gx_flux_raw = Flux.gradient(loss_flux, x_flux)[1]

  # z powrotem do (C,H,W)
  y_flux = dropdims(y_flux_raw; dims=4)
  y_flux = permutedims(y_flux, (3, 1, 2))

  gx_flux = dropdims(gx_flux_raw; dims=4)
  gx_flux = permutedims(gx_flux, (3, 1, 2))

  println("=== OUTPUT ===")
  println("size(y_my)              = ", size(y_my))
  println("size(y_flux)            = ", size(y_flux))
  println("max |y_my - y_flux|     = ", maximum(abs.(y_my .- y_flux)))

  println("\n=== INPUT GRAD ===")
  println("size(gx_my)             = ", size(gx_my))
  println("size(gx_flux)           = ", size(gx_flux))
  println("max |dx_my - dx_flux|   = ", maximum(abs.(gx_my .- gx_flux)))

  println("\n=== FULL VALUES ===")
  println("y_my = ")
  display(y_my)
  println("y_flux = ")
  display(y_flux)

  println("gx_my = ")
  display(gx_my)
  println("gx_flux = ")
  display(gx_flux)

  return (
    x=xval,
    y_my=y_my,
    y_flux=y_flux,
    gx_my=gx_my,
    gx_flux=gx_flux,
  )
end

function compare_flatten_after_maxpool_my_vs_flux(; C=2, H=4, W=4, kernel=(2, 2), seed=1234, T=Float64)
  Random.seed!(seed)

  kh, kw = kernel
  @assert H % kh == 0
  @assert W % kw == 0

  # wejście bez remisów w maxpool
  xval = randn(T, C, H, W)
  xval .+= reshape(T.(1:length(xval)) .* T(1e-8), size(xval))

  # ===== Mój framework =====
  input = tensor(C, H, W)
  pool_node = maxpool(kernel)(input)
  flat_node = flatten()(pool_node)

  model_my = graph(flat_node)

  zerograd!(model_my)
  forward!(model_my, input => xval)
  backward!(model_my)   # seed.grad .= 1 => loss = sum(output)

  y_my = copy(flat_node.data)
  gx_my = copy(input.grad)

  # ===== Flux =====
  # moje: (C,H,W)
  # Flux: (H,W,C,N)
  x_flux = permutedims(xval, (2, 3, 1))
  x_flux = reshape(x_flux, H, W, C, 1)

  pool = Flux.MaxPool(kernel)

  function flux_model(x)
    pooled = pool(x)                 # (H', W', C, 1)
    return vec(pooled)               # flatten całego tensora do 1D
  end

  loss_flux(x) = sum(flux_model(x))

  y_flux = flux_model(x_flux)
  gx_flux_raw = Flux.gradient(loss_flux, x_flux)[1]

  # gradient z powrotem do (C,H,W)
  gx_flux = dropdims(gx_flux_raw; dims=4)
  gx_flux = permutedims(gx_flux, (3, 1, 2))

  println("=== OUTPUT ===")
  println("size(y_my)              = ", size(y_my))
  println("size(y_flux)            = ", size(y_flux))
  println("max |y_my - y_flux|     = ", maximum(abs.(y_my .- y_flux)))

  println("\n=== INPUT GRAD ===")
  println("size(gx_my)             = ", size(gx_my))
  println("size(gx_flux)           = ", size(gx_flux))
  println("max |dx_my - dx_flux|   = ", maximum(abs.(gx_my .- gx_flux)))

  println("\n=== FULL VALUES ===")
  println("y_my = ")
  display(y_my)
  println("y_flux = ")
  display(y_flux)

  println("gx_my = ")
  display(gx_my)
  println("gx_flux = ")
  display(gx_flux)

  return (
    x=xval,
    y_my=y_my,
    y_flux=y_flux,
    gx_my=gx_my,
    gx_flux=gx_flux,
  )
end
function compare_conv_my_vs_flux(;
  in_channels=1,
  out_channels=1,
  H=4,
  W=4,
  kernel=(3, 3),
  pad=1,
  seed=1234,
  T=Float64,
)
  Random.seed!(seed)

  kh, kw = kernel
  xval = randn(T, in_channels, H, W)

  # ===== Mój framework =====
  input = tensor(in_channels, H, W)
  conv_node = conv(kernel, in_channels => out_channels, pad=pad, bias=false)(input)

  kernels_node, _, _ = conv_node.args

  W_my_val = randn(T, out_channels, in_channels, kh, kw)
  kernels_node.data .= W_my_val

  model_my = graph(conv_node)

  zerograd!(model_my)
  forward!(model_my, input => xval)
  backward!(model_my)   # seed.grad .= 1 => loss = sum(output)

  y_my = copy(conv_node.data)
  gx_my = copy(input.grad)
  gW_my = copy(kernels_node.grad)

  # ===== Flux =====
  # moje: (C,H,W)
  # Flux: (W,H,C,N)
  x_flux = permutedims(xval, (3, 2, 1))
  x_flux = reshape(x_flux, W, H, in_channels, 1)

  layer = Flux.Conv(kernel, in_channels => out_channels; pad=pad, bias=false)
  layer = Flux.f64(layer)

  # moje wagi: (OC, IC, KH, KW)
  # Flux wagi: (KW, KH, IC, OC)
  #
  # Dodatkowo obracamy kernel o 180 stopni, żeby dopasować konwencję
  W_flip = reverse(reverse(W_my_val; dims=3); dims=4)
  W_flux_val = permutedims(W_flip, (4, 3, 2, 1))
  layer.weight .= W_flux_val

  loss_flux(m, x) = sum(m(x))

  y_flux_raw = layer(x_flux)
  gx_flux_raw = Flux.gradient(x -> loss_flux(layer, x), x_flux)[1]
  g_layer = Flux.gradient(m -> loss_flux(m, x_flux), layer)[1]
  gW_flux_raw = g_layer.weight

  # output: Flux (Wout,Hout,OC,1) -> moje (OC,Hout,Wout)
  y_flux = dropdims(y_flux_raw; dims=4)
  y_flux = permutedims(y_flux, (3, 2, 1))

  # input grad: Flux (W,H,IC,1) -> moje (IC,H,W)
  gx_flux = dropdims(gx_flux_raw; dims=4)
  gx_flux = permutedims(gx_flux, (3, 2, 1))

  # weight grad: Flux (KW,KH,IC,OC) -> moje (OC,IC,KH,KW)
  gW_flux = permutedims(gW_flux_raw, (4, 3, 2, 1))

  # cofamy obrót kernela, żeby porównać w Twojej konwencji
  gW_flux = reverse(reverse(gW_flux; dims=3); dims=4)

  println("=== OUTPUT ===")
  println("size(y_my)              = ", size(y_my))
  println("size(y_flux)            = ", size(y_flux))
  println("max |y_my - y_flux|     = ", maximum(abs.(y_my .- y_flux)))

  println("\n=== INPUT GRAD ===")
  println("size(gx_my)             = ", size(gx_my))
  println("size(gx_flux)           = ", size(gx_flux))
  println("max |dx_my - dx_flux|   = ", maximum(abs.(gx_my .- gx_flux)))

  println("\n=== WEIGHT GRAD ===")
  println("size(gW_my)             = ", size(gW_my))
  println("size(gW_flux)           = ", size(gW_flux))
  println("max |dW_my - dW_flux|   = ", maximum(abs.(gW_my .- gW_flux)))

  return (
    y_my=y_my,
    y_flux=y_flux,
    gx_my=gx_my,
    gx_flux=gx_flux,
    gW_my=gW_my,
    gW_flux=gW_flux,
  )
end


# compare_dense_my_vs_flux()
# compare_maxpool_my_vs_flux()
# compare_flatten_after_maxpool_my_vs_flux()
# compare_conv_my_vs_flux()

function compare_small_conv_flatten_dense_my_vs_flux(; seed=1234, T=Float64)
  Random.seed!(seed)

  # =========================================
  # Ustawienia małej sieci
  # =========================================
  C, H, W = 1, 4, 4
  OC = 1
  KH, KW = 3, 3
  PAD = 1
  OUT = 3

  xval = randn(T, C, H, W)

  # =========================================
  # Mój framework
  # =========================================
  input = tensor(C, H, W)

  conv_node = conv((KH, KW), C => OC, pad=PAD, bias=false)(input)
  flat_node = flatten()(conv_node)
  dense_node = dense(length(flat_node.data) => OUT)(flat_node)

  model_my = graph(dense_node)

  # wyciąganie parametrów
  K_node, _, _ = conv_node.args

  mul_node, b_node = dense_node.args
  W_node, _ = mul_node.args

  # =========================================
  # Wspólne wagi
  # =========================================
  Kval = randn(T, OC, C, KH, KW)
  Wval = randn(T, OUT, OC * H * W)
  bval = randn(T, OUT)

  K_node.data .= Kval
  W_node.data .= Wval
  b_node.data .= bval

  # =========================================
  # Forward/backward - mój framework
  # =========================================
  zerograd!(model_my)
  forward!(model_my, input => xval)
  backward!(model_my)   # seed = ones(output), czyli loss = sum(output)

  y_my = copy(dense_node.data)
  gx_my = copy(input.grad)
  gK_my = copy(K_node.grad)
  gW_my = copy(W_node.grad)
  gb_my = copy(b_node.grad)

  # =========================================
  # Flux
  # =========================================
  # moje wejście: (C,H,W)
  # Flux conv:    (W,H,C,N)
  x_flux = permutedims(xval, (3, 2, 1))
  x_flux = reshape(x_flux, W, H, C, 1)

  conv_flux = Flux.Conv((KH, KW), C => OC; pad=PAD, bias=false) |> Flux.f64
  dense_flux = Flux.Dense(OC * H * W => OUT) |> Flux.f64

  # moje K:   (OC, IC, KH, KW)
  # Flux K:   (KW, KH, IC, OC)
  K_flip = reverse(reverse(Kval; dims=3); dims=4)
  K_flux_val = permutedims(K_flip, (4, 3, 2, 1))

  conv_flux.weight .= K_flux_val
  dense_flux.weight .= Wval
  dense_flux.bias .= bval

  function flux_model(x)
    y = conv_flux(x)         # (W,H,OC,N)
    y = vec(y)               # flatten całego tensora
    y = dense_flux(y)
    return y
  end

  loss_flux_x(x) = sum(flux_model(x))

  y_flux = flux_model(x_flux)

  gx_flux_raw = Flux.gradient(loss_flux_x, x_flux)[1]

  g_conv = Flux.gradient(c -> sum(dense_flux(vec(c(x_flux)))), conv_flux)[1]
  g_dense = Flux.gradient(d -> sum(d(vec(conv_flux(x_flux)))), dense_flux)[1]

  # =========================================
  # Konwersje z Flux do mojej konwencji
  # =========================================
  gx_flux = dropdims(gx_flux_raw; dims=4)
  gx_flux = permutedims(gx_flux, (3, 2, 1))

  gK_flux = permutedims(g_conv.weight, (4, 3, 2, 1))
  gK_flux = reverse(reverse(gK_flux; dims=3); dims=4)

  gW_flux = g_dense.weight
  gb_flux = g_dense.bias

  # =========================================
  # Raport
  # =========================================
  println("=== OUTPUT ===")
  println("size(y_my)              = ", size(y_my))
  println("size(y_flux)            = ", size(y_flux))
  println("max |y_my - y_flux|     = ", maximum(abs.(vec(y_my) .- vec(y_flux))))

  println("\n=== INPUT GRAD ===")
  println("size(gx_my)             = ", size(gx_my))
  println("size(gx_flux)           = ", size(gx_flux))
  println("max |dx_my - dx_flux|   = ", maximum(abs.(gx_my .- gx_flux)))

  println("\n=== CONV WEIGHT GRAD ===")
  println("size(gK_my)             = ", size(gK_my))
  println("size(gK_flux)           = ", size(gK_flux))
  println("max |dK_my - dK_flux|   = ", maximum(abs.(gK_my .- gK_flux)))

  println("\n=== DENSE WEIGHT GRAD ===")
  println("size(gW_my)             = ", size(gW_my))
  println("size(gW_flux)           = ", size(gW_flux))
  println("max |dW_my - dW_flux|   = ", maximum(abs.(gW_my .- gW_flux)))

  println("\n=== DENSE BIAS GRAD ===")
  println("size(gb_my)             = ", size(gb_my))
  println("size(gb_flux)           = ", size(gb_flux))
  println("max |db_my - db_flux|   = ", maximum(abs.(gb_my .- gb_flux)))

  println("=== BEFORE STEP ===")
  println("sum(K) = ", sum(K_node.data))
  println("sum(W) = ", sum(W_node.data))
  println("sum(b) = ", sum(b_node.data))

  println("sum(abs(gK)) = ", sum(abs.(K_node.grad)))
  println("sum(abs(gW)) = ", sum(abs.(W_node.grad)))
  println("sum(abs(gb)) = ", sum(abs.(b_node.grad)))

  K_before = copy(K_node.data)
  W_before = copy(W_node.data)
  b_before = copy(b_node.data)

  lr = 1e-2
  optimize!(model_my, lr)

  println("\n=== AFTER STEP ===")
  println("max |ΔK| = ", maximum(abs.(K_node.data .- K_before)))
  println("max |ΔW| = ", maximum(abs.(W_node.data .- W_before)))
  println("max |Δb| = ", maximum(abs.(b_node.data .- b_before)))

  return (
    x=xval,
    y_my=y_my,
    y_flux=y_flux,
    gx_my=gx_my,
    gx_flux=gx_flux,
    gK_my=gK_my,
    gK_flux=gK_flux,
    gW_my=gW_my,
    gW_flux=gW_flux,
    gb_my=gb_my,
    gb_flux=gb_flux,
  )
end

# compare_small_conv_flatten_dense_my_vs_flux()
# compare_conv_my_vs_flux()
compare_flatten_after_maxpool_my_vs_flux()

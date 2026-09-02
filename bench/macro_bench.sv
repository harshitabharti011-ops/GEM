// Scaled macro benchmark.
//
// macro_smoke.sv proves correctness on 7 macros. It cannot show throughput,
// because at 102 cells it maps to a SINGLE boomerang partition either way --
// and GEM's script size scales with PARTITIONS, not with gate count. A 104x
// reduction in AIG cells therefore shrank the script by only 26%, leaving the
// run dominated by grid-barrier latency rather than by memory bandwidth.
//
// This design exists to move the workload into the regime the optimisation
// actually targets. Three properties matter, and each is deliberate:
//
//   1. MANY ENDPOINTS. Partitions come from endpoints, not gates. Every lane
//      carries a 48-bit registered accumulator, so LANES=64 gives 3072 flop
//      endpoints and forces the design across many partitions. That is what
//      makes the shredded script large enough to saturate DRAM.
//
//   2. MACRO -> BOOLEAN. flags[l] compares a CARRY4 sum against a constant.
//      That is exactly the dependency that was silently broken (a macro
//      result reaching AND-gate logic), so the benchmark regression-tests the
//      fix rather than merely exercising macros in isolation.
//
//   3. MACRO -> MACRO. Each lane's carry chain feeds CO[3] into the next
//      block's CIN, the dependency the problem statement names explicitly.
//
// Lanes are perturbed by their index so Yosys cannot common-subexpression
// them into one lane. Outputs are XOR-reductions so every lane stays
// observable -- an unobservable lane would be optimised away and the gate
// count would silently collapse.
//
// Style note: this deliberately avoids SystemVerilog casts, unpacked arrays
// in port connections and empty port connections. All three are legal, and
// all three are places where synthesiser frontends differ. Everything here is
// Verilog-2001-shaped so the two synthesis flows cannot disagree about it.
//
// Sweep with bench/gen_bench.sh; LANES is the only knob.

module macro_bench #(
    parameter LANES = 16
) (
    input  wire               clk,
    input  wire signed [26:0] a_in, d_in,
    input  wire signed [17:0] b_in,
    input  wire signed [47:0] c_in,
    input  wire        [1:0]  mode,
    input  wire        [15:0] x_in, y_in,
    input  wire        [4:0]  tap,
    input  wire               ser_in, ser_en,
    output reg  signed [47:0] acc_xor,
    output reg         [16:0] sum_xor,
    output wire [LANES-1:0]   flags,
    output reg                delayed_xor
);

  // Flattened per-lane results. Packed, not unpacked: an unpacked array
  // element in an instance port connection is the kind of thing frontends
  // disagree about.
  wire [48*LANES-1:0] acc_flat;
  wire [17*LANES-1:0] sum_flat;
  wire    [LANES-1:0] delayed;

  genvar l, i;
  generate
    for (l = 0; l < LANES; l = l + 1) begin : g_lane

      // --- per-lane perturbation ------------------------------------------
      // Without this every lane is structurally identical and Yosys folds
      // them into one, which would defeat the whole point of scaling.
      localparam [26:0] PA = l;
      localparam [26:0] PD = l * 3;
      localparam [17:0] PB = l * 5;
      localparam [47:0] PC = l * 7;
      localparam [15:0] PX = l * 11;
      localparam [15:0] PY = l * 13;
      localparam [4:0]  PT = l;

      wire signed [26:0] a_l = a_in ^ PA;
      wire signed [26:0] d_l = d_in ^ PD;
      wire signed [17:0] b_l = b_in ^ PB;
      wire signed [47:0] c_l = c_in ^ PC;
      wire        [15:0] x_l = x_in ^ PX;
      wire        [15:0] y_l = y_in ^ PY;

      // --- MAC ------------------------------------------------------------
      wire signed [47:0] p_l;
      DSP48E2 u_dsp (.CLK(clk), .A(a_l), .B(b_l), .C(c_l), .D(d_l),
                     .USE_PREADD(1'b1), .MODE(mode), .P(p_l));

      // A 48-bit accumulator per lane. This is the endpoint multiplier: it is
      // what turns a small design into a many-partition one.
      reg signed [47:0] acc_r;
      always @(posedge clk) acc_r <= acc_r ^ p_l;
      assign acc_flat[48*l +: 48] = acc_r;

      // --- 16-bit adder as a real CARRY4 chain ----------------------------
      wire [16:0] sum_l;
      wire [15:0] prop = x_l ^ y_l;
      wire [4:0]  carry;
      assign carry[0] = 1'b0;
      for (i = 0; i < 4; i = i + 1) begin : g_carry
        wire [3:0] co_i;
        CARRY4 u_c4 (.S(prop[i*4 +: 4]), .DI(x_l[i*4 +: 4]),
                     .CIN(carry[i]), .CYINIT(1'b0),
                     .CO(co_i), .O(sum_l[i*4 +: 4]));
        assign carry[i+1] = co_i[3];
      end
      assign sum_l[16] = carry[4];
      assign sum_flat[17*l +: 17] = sum_l;

      // --- MACRO -> BOOLEAN: the dependency that was broken ---------------
      // A comparator on the CARRY4 result. Ordinary AND-gate logic consuming
      // a word-level macro output.
      assign flags[l] = (sum_l > 17'd100);

      // --- delay line -----------------------------------------------------
      wire q31_unused;
      SRLC32E u_srl (.CLK(clk), .D(ser_in ^ PA[0]), .CE(ser_en),
                     .A(tap ^ PT), .Q(delayed[l]), .Q31(q31_unused));
    end
  endgenerate

  // XOR-reduce so every lane remains observable at a top-level port without
  // needing LANES x 48 primary outputs.
  integer k;
  always @* begin
    acc_xor     = 48'd0;
    sum_xor     = 17'd0;
    delayed_xor = 1'b0;
    for (k = 0; k < LANES; k = k + 1) begin
      acc_xor     = acc_xor ^ acc_flat[48*k +: 48];
      sum_xor     = sum_xor ^ sum_flat[17*k +: 17];
      delayed_xor = delayed_xor ^ delayed[k];
    end
  end

endmodule

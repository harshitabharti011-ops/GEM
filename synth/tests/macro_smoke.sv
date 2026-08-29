// Smoke test for the macro-preserving synthesis flow.
// Deliberately mixes all three macros with ordinary RTL that SHOULD be shredded
// into AIG cells, plus a DFF and a small RAM so step 1 is exercised too.
//
//   synth/run_synth.sh synth/tests/macro_smoke.sv macro_smoke
//
// Expected: 2 x $__GEMDSP_, 4 x $__GEMCARRY4_, 1 x $__GEMSRL32_ survive,
// surrounded by a few hundred AND2/INV/DFF cells from the ordinary logic.

module macro_smoke (
    input  logic               clk,
    input  logic signed [26:0] a0, a1, d0,
    input  logic signed [17:0] b0, b1,
    input  logic signed [47:0] c0,
    input  logic        [1:0]  mode,
    input  logic        [15:0] x, y,
    input  logic        [4:0]  tap,
    input  logic               ser_in, ser_en,
    output logic signed [47:0] acc0, acc1,
    output logic        [16:0] sum,
    output logic               delayed, delayed31,
    output logic        [7:0]  scratch
);

  // --- two DSPs, one accumulating, one multiply-only ---------------------
  DSP48E2 u_dsp0 (.CLK(clk), .A(a0), .B(b0), .C(c0), .D(d0),
                  .USE_PREADD(1'b1), .MODE(mode), .P(acc0));
  DSP48E2 u_dsp1 (.CLK(clk), .A(a1), .B(b1), .C(48'sd0), .D(27'sd0),
                  .USE_PREADD(1'b0), .MODE(2'd1), .P(acc1));

  // --- a 16-bit adder built as a real CARRY4 chain -----------------------
  // CO[3] of each stage feeds CIN of the next: exactly the macro-to-macro
  // dependency the PS calls out in deliverable B.
  wire [15:0] p = x ^ y;          // propagate
  wire [4:0]  carry;
  assign carry[0] = 1'b0;
  genvar i;
  generate for (i = 0; i < 4; i++) begin : g_carry
     wire [3:0] co_i;
     CARRY4 u_c4 (.S(p[i*4 +: 4]), .DI(x[i*4 +: 4]),
                  .CIN(carry[i]), .CYINIT(1'b0),
                  .CO(co_i), .O(sum[i*4 +: 4]));
     assign carry[i+1] = co_i[3];
  end endgenerate
  assign sum[16] = carry[4];

  // --- a shift-register delay line ---------------------------------------
  SRLC32E u_srl (.CLK(clk), .D(ser_in), .CE(ser_en), .A(tap),
                 .Q(delayed), .Q31(delayed31));

  // --- ordinary logic: this SHOULD become AIG cells ----------------------
  logic [7:0] cnt;
  always_ff @(posedge clk) cnt <= cnt + 8'd1;
  assign scratch = cnt ^ {x[3:0], y[3:0]};
endmodule

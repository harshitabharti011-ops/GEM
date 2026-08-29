// SWAP FILE -- the hedge.
//
// Maps whatever the incoming SystemVerilog calls its primitives onto GEM's
// canonical macro cells in synth/gem_macros.v. Read as a NORMAL module (not
// -lib) BEFORE the design; `hierarchy` + `flatten` resolve straight through it,
// leaving only $__GEMDSP_ / $__GEMCARRY4_ / $__GEMSRL32_ instances behind.
//
// Currently: the PS's simplified subset, near-identity.
// If the hidden benchmarks instantiate real Xilinx primitives, replace this
// file with synth/macro_normalize_xilinx.v. Nothing downstream moves.

module DSP48E2 (CLK, A, B, C, D, USE_PREADD, MODE, P);
   input               CLK, USE_PREADD;
   input signed [26:0] A, D;
   input signed [17:0] B;
   input signed [47:0] C;
   input        [1:0]  MODE;
   output signed [47:0] P;

   \$__GEMDSP_ _TECHMAP_REPLACE_ (
      .CLK(CLK), .A(A), .B(B), .C(C), .D(D),
      .USE_PREADD(USE_PREADD), .MODE(MODE), .P(P));
endmodule

module CARRY4 (S, DI, CIN, CYINIT, CO, O);
   input  [3:0] S, DI;
   input        CIN, CYINIT;
   output [3:0] CO, O;

   \$__GEMCARRY4_ _TECHMAP_REPLACE_ (
      .S(S), .DI(DI), .CIN(CIN), .CYINIT(CYINIT), .CO(CO), .O(O));
endmodule

module SRLC32E (CLK, D, CE, A, Q, Q31);
   input        CLK, D, CE;
   input  [4:0] A;
   output       Q, Q31;

   \$__GEMSRL32_ _TECHMAP_REPLACE_ (
      .CLK(CLK), .D(D), .CE(CE), .A(A), .Q(Q), .Q31(Q31));
endmodule

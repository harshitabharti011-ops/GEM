// ALTERNATIVE swap file -- use only if the hidden benchmarks instantiate real
// Xilinx primitives. Copy over synth/macro_normalize.v; change nothing else.
//
//     cp synth/macro_normalize_xilinx.v synth/macro_normalize.v
//
// This is where the PS's "Yosys parser must extract the intent and pass a
// simplified 2-bit state" actually happens: OPMODE[8:0] -> MODE[1:0].
// Yosys `opt_expr` constant-folds it away when OPMODE is tied off, which it is
// in essentially all real netlists.

module DSP48E2 (CLK, A, B, C, D, OPMODE, INMODE, ALUMODE, P);
   input                CLK;
   input signed  [29:0] A;        // real DSP48E2 is A[29:0]; we use A[26:0]
   input signed  [17:0] B;
   input signed  [47:0] C;
   input signed  [26:0] D;
   input         [8:0]  OPMODE;   // {W[8:7], Z[6:4], Y[3:2], X[1:0]}
   input         [4:0]  INMODE;
   input         [3:0]  ALUMODE;
   output signed [47:0] P;

   // Z field selects the accumulator source:
   //   Z=010 -> P   (multiply-accumulate)
   //   Z=011 -> C
   //   Z=000 -> 0
   // X/Y = 01/01 routes the multiplier partial products into the ALU.
   wire mult_in = (OPMODE[3:0] == 4'b0101);
   wire [1:0] mode = (OPMODE[6:4] == 3'b010) ? 2'd2 :   // accumulate
                     (mult_in)               ? 2'd1 :   // multiply only
                                               2'd0;    // bypass -> C
   wire preadd = INMODE[0];       // INMODE[0]=1 selects the pre-adder path

   \$__GEMDSP_ _TECHMAP_REPLACE_ (
      .CLK(CLK), .A(A[26:0]), .B(B), .C(C), .D(D),
      .USE_PREADD(preadd), .MODE(mode), .P(P));
endmodule

module CARRY4 (CO, O, CI, CYINIT, DI, S);
   output [3:0] CO, O;
   input        CI, CYINIT;       // note: real Xilinx spells it CI, PS says CIN
   input  [3:0] DI, S;

   \$__GEMCARRY4_ _TECHMAP_REPLACE_ (
      .S(S), .DI(DI), .CIN(CI), .CYINIT(CYINIT), .CO(CO), .O(O));
endmodule

module SRLC32E (Q, Q31, A, CE, CLK, D);
   output       Q, Q31;
   input  [4:0] A;
   input        CE, CLK, D;

   \$__GEMSRL32_ _TECHMAP_REPLACE_ (
      .CLK(CLK), .D(D), .CE(CE), .A(A), .Q(Q), .Q31(Q31));
endmodule

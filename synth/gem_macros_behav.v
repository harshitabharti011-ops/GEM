// Simulation-only bodies for the canonical macro cells.
// NOT read during synthesis. Use this to simulate gatelevel.gv in iverilog /
// Verilator, exactly as usage.md tells you to check step 2.
//
// This file is also the spec that csrc/macros.cuh must match bit for bit --
// it is deliverable C's golden model in Verilog form.
// Every equation below is transcribed directly from the problem statement.

module \$__GEMDSP_ (CLK, A, B, C, D, USE_PREADD, MODE, P);
   input                CLK, USE_PREADD;
   input signed  [26:0] A, D;
   input signed  [17:0] B;
   input signed  [47:0] C;
   input         [1:0]  MODE;
   output reg signed [47:0] P;

   // AREG/BREG/CREG/DREG/ADREG/MREG are all combinational (PS: set to 0).
   // AD is a 27-bit pre-adder output and WRAPS -- real DSP48E2 silicon has a
   // 27-bit pre-adder, which is why the PS specifies a 45-bit product. Widening
   // AD to 28 bits changes every result once the pre-adder overflows.
   wire signed [26:0] AD = USE_PREADD ? (A + D) : A;
   wire signed [44:0] M  = AD * B;

   initial P = 48'sd0;                       // PS note 3: registers init to zero
   always @(posedge CLK)                     // PS note 2: single global clock
     case (MODE)
       2'd0:    P <= C;                      // state 0: bypass
       2'd1:    P <= $signed(M);             // state 1: multiply only
       default: P <= P + $signed(M);         // state 2: multiply-accumulate
     endcase
endmodule

module \$__GEMCARRY4_ (S, DI, CIN, CYINIT, CO, O);
   input  [3:0] S, DI;
   input        CIN, CYINIT;
   output [3:0] CO, O;

   wire [4:0] Cv;
   assign Cv[0] = CYINIT | CIN;              // only one is active in valid RTL
   genvar i;
   generate for (i = 0; i < 4; i = i + 1) begin : chain
      assign Cv[i+1] = (S[i] & Cv[i]) | (~S[i] & DI[i]);
      assign O[i]    = S[i] ^ Cv[i];
      assign CO[i]   = Cv[i+1];
   end endgenerate
endmodule

module \$__GEMSRL32_ (CLK, D, CE, A, Q, Q31);
   input        CLK, D, CE;
   input  [4:0] A;
   output       Q, Q31;

   reg [31:0] sr;
   initial sr = 32'b0;                       // PS note 3
   always @(posedge CLK) if (CE) sr <= {sr[30:0], D};   // shift LSB -> MSB
   assign Q   = sr[A];                       // combinational dynamic read
   assign Q31 = sr[31];                      // combinational cascade out
endmodule

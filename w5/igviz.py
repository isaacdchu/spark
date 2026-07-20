from manim import *
import numpy as np

# Visualize implicit GEMM NHWC/KRSC convolution using Manim.
#
# Mirrors conv2d_implicit_gemm_krsc in src/conv.hpp:
#   output[n,p,q,k] = sum_r sum_s sum_c  input[n, p+r, q+s, c] * kernel[k, r, s, c]
# (padding/stride/dilation are all trivial here, so h_in=p+r, w_in=q+s).
#
# As an implicit GEMM this is  A[M, RSC] . B[RSC, K] = C[M, K]  with
#   M = P*Q   (output pixels)      -> rows of A / C
#   N = K     (output channels)    -> cols of B / C
#   L = R*S*C (flattened field)    -> shared reduction axis
#
# No tensor *values* are ever shown: identity is carried entirely by color,
# and the multiply/accumulate is conveyed by motion.

# ---------------------------------------------------------------- dimensions
H = W = 4        # input spatial
R = S = 2        # kernel spatial
C = 3            # channels (the contiguous / reduction inner axis)
K = 2            # output channels (number of filters)
P = Q = H - R + 1  # output spatial = 3
M = P * Q          # 9  GEMM rows
L = R * S * C      # 12 GEMM reduction length

# ---------------------------------------------------------------- palette
FILTER_HUES = [ManimColor(ORANGE), ManimColor("#E388BB")]  # warm hue per filter
EMPTY_FILL = ManimColor("#7b7b7b")     # unfilled output / C-matrix cell
WINDOW_COLOR = YELLOW
DEPTH = np.array([0.1, 0.1, 0.0])  # per-channel diagonal offset (fake 3D)


def input_base_color(h, w):
    """Stable positional identity for input pixel (h, w): teal -> blue sweep."""
    t = (h * W + w) / (H * W - 1)
    return interpolate_color(TEAL_E, BLUE_D, t)


def shade(color, c, n=C):
    """Darken by channel index so the C axis reads as depth (front = brightest)."""
    return interpolate_color(ManimColor(color), BLACK, 0.42 * c / max(n - 1, 1))


def blend(a, b):
    return interpolate_color(ManimColor(a), ManimColor(b), 0.7)


def patch_avg_color(p, q):
    cols = [input_base_color(p + r, q + s) for r in range(R) for s in range(S)]
    acc = np.mean([color_to_rgb(c) for c in cols], axis=0)
    return rgb_to_color(acc)


def channel_stack(base_color, n_channels=C, side=0.5):
    """A single spatial cell drawn as n offset cards => the channel/depth axis."""
    cards = VGroup()
    for c in range(n_channels):
        sq = Square(side_length=side)
        sq.set_fill(shade(base_color, c, n_channels), opacity=1.0)
        sq.set_stroke(WHITE, width=1.2)
        sq.shift(DEPTH * c)          # deeper channels up-right...
        sq.set_z_index(n_channels - c)  # ...and behind
        cards.add(sq)
    return cards


def spatial_grid(nrows, ncols, color_fn, n_channels=C, side=0.5, gap=0.16):
    """Return (VGroup, cells[r][c]) laid out as an nrows x ncols grid of stacks."""
    cells = [[None] * ncols for _ in range(nrows)]
    group = VGroup()
    step = side + gap
    for r in range(nrows):
        for c in range(ncols):
            st = channel_stack(color_fn(r, c), n_channels, side)
            st.move_to([c * step, -r * step, 0])
            cells[r][c] = st
            group.add(st)
    group.center()
    return group, cells


def matrix_grid(nrows, ncols, color_fn, side=0.2, gap=0.03):
    """Flat grid of unit squares (for the GEMM matrices). cells[r][c] = square."""
    cells = [[None] * ncols for _ in range(nrows)]
    group = VGroup()
    step = side + gap
    for r in range(nrows):
        for c in range(ncols):
            col, op = color_fn(r, c)
            sq = Square(side_length=side)
            sq.set_fill(col, opacity=op)
            sq.set_stroke(WHITE, width=0.8)
            sq.move_to([c * step, -r * step, 0])
            cells[r][c] = sq
            group.add(sq)
    group.center()
    return group, cells


class ImplicitGemmConv(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e20"
        self.part1_spatial()
        self.part2_gemm()

    # ============================================================= PART 1
    def part1_spatial(self):
        title = Text("Convolution: NHWC input ⊛ KRSC kernel", weight=BOLD).scale(0.6)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.2))

        # --- input tensor (left) --------------------------------------------
        inp_grid, inp = spatial_grid(H, W, input_base_color)
        inp_grid.scale(0.82).move_to([-4.7, 0.2, 0])
        inp_label = Text("input (H×W×C)").scale(0.4)
        inp_label.next_to(inp_grid, DOWN, buff=0.35)

        # --- kernel filters (center) ----------------------------------------
        filt = [[[None] * S for _ in range(R)] for _ in range(K)]
        filt_groups = VGroup()
        for k in range(K):
            g, cells = spatial_grid(R, S, lambda r, c, k=k: FILTER_HUES[k])
            g.scale(0.82)
            filt[k] = cells
            filt_groups.add(g)
        filt_groups.arrange(DOWN, buff=0.7).move_to([-0.7, 0.2, 0])
        filt_labels = VGroup(*[
            Text(f"filter {k}").scale(0.36).next_to(filt_groups[k], LEFT, buff=0.25)
            for k in range(K)
        ])
        kernel_label = Text("kernel: K filters of R×S×C").scale(0.4)
        kernel_label.next_to(filt_groups, DOWN, buff=0.35)

        # --- output tensor (right) ------------------------------------------
        out_grid, out = spatial_grid(P, Q, lambda r, c: EMPTY_FILL, n_channels=K)
        out_grid.scale(0.82).move_to([4.6, 0.2, 0])
        for row in out:
            for st in row:
                for sq in st:
                    sq.set_fill(EMPTY_FILL, opacity=0.55)
        out_label = Text("output (P×Q×K)").scale(0.4)
        out_label.next_to(out_grid, DOWN, buff=0.35)

        self.play(
            LaggedStart(
                FadeIn(inp_grid, scale=0.9), Write(inp_label),
                FadeIn(filt_groups, scale=0.9), *[Write(l) for l in filt_labels],
                Write(kernel_label),
                FadeIn(out_grid, scale=0.9), Write(out_label),
                lag_ratio=0.15,
            )
        )
        self.wait(0.3)

        # channel-axis annotation on the first input cell
        c_arrow = Arrow(
            inp[0][0][0].get_center(),
            inp[0][0][0].get_center() + DEPTH * 10,
            buff=0, color=WHITE, stroke_width=3, max_tip_length_to_length_ratio=0.3,
        )
        c_tag = MathTex("C").scale(0.6).next_to(c_arrow, UR, buff=0.05)
        self.play(GrowArrow(c_arrow), FadeIn(c_tag))
        self.play(
            LaggedStart(*[Indicate(inp[0][0][c], color=WHITE, scale_factor=1.25)
                          for c in range(C)], lag_ratio=0.5)
        )
        self.play(FadeOut(c_arrow), FadeOut(c_tag))

        # receptive-field window over input patch (p, q) = (0, 0)
        def patch_rect(p, q):
            stacks = VGroup(*[inp[p + r][q + s] for r in range(R) for s in range(S)])
            rect = SurroundingRectangle(stacks, color=WINDOW_COLOR, buff=0.06)
            rect.set_z_index(C + 1)  # draw in front of the channel cards
            return rect

        window = patch_rect(0, 0)
        self.play(Create(window))

        # ---- full-detail dot products for output pixel (0,0), both filters ----
        for k in range(K):
            self.dot_product(inp, filt, out, 0, 0, k, detail=(k == 0))
        self.wait(0.3)

        # ---- one more position at medium detail: (0,1) ----
        self.play(window.animate.move_to(patch_rect(0, 1).get_center()))
        for k in range(K):
            self.dot_product(inp, filt, out, 0, 1, k, detail=False, quick=True)

        # ---- fast-forward: sweep the rest ----
        remaining = [(p, q) for p in range(P) for q in range(Q) if (p, q) not in [(0, 0), (0, 1)]]
        for p, q in remaining:
            self.play(window.animate.move_to(patch_rect(p, q).get_center()), run_time=0.28)
            fills = []
            for k in range(K):
                col = blend(patch_avg_color(p, q), FILTER_HUES[k])
                fills.append(out[p][q][k].animate.set_fill(shade(col, k, K), opacity=1.0))
            self.play(LaggedStart(*fills, lag_ratio=0.25), run_time=0.4)
        self.wait(0.5)

        # clear text; keep the tensors on screen so part 2 can morph them
        self.play(FadeOut(window), FadeOut(VGroup(
            title, inp_label, filt_labels, kernel_label, out_label,
        )))
        self.inp_grid, self.inp = inp_grid, inp
        self.filt_groups, self.filt = filt_groups, filt
        self.out_grid, self.out = out_grid, out

    def dot_product(self, inp, filt, out, p, q, k, detail=False, quick=False):
        """Animate the R*S*C multiply-accumulate feeding output pixel (p,q,k)."""
        hue = FILTER_HUES[k]
        lines, pulses = VGroup(), []
        for r in range(R):
            for s in range(S):
                a = inp[p + r][q + s]
                b = filt[k][r][s]
                lines.add(Line(a[0].get_center(), b[0].get_center(),
                               color=hue, stroke_width=2.5, stroke_opacity=0.9))
                pulses += [Indicate(a, color=hue, scale_factor=1.15),
                           Indicate(b, color=WHITE, scale_factor=1.15)]

        rt = 0.5 if quick else 1.0
        self.play(LaggedStartMap(Create, lines, lag_ratio=0.12), run_time=rt)

        if detail:
            # foreground the C reduction on the first tap only
            a0 = inp[p][q]
            c_lbl = MathTex(r"\sum_c").scale(0.5).next_to(a0, UP, buff=0.15).set_color(hue)
            self.play(FadeIn(c_lbl))
            self.play(LaggedStart(*[Indicate(a0[c], color=hue, scale_factor=1.3)
                                    for c in range(C)], lag_ratio=0.4))
            self.play(FadeOut(c_lbl))
            self.play(LaggedStart(*pulses, lag_ratio=0.08), run_time=1.0)
        else:
            self.play(LaggedStart(*pulses, lag_ratio=0.05), run_time=rt)

        # accumulate: per-tap product copies fly next to the output cell as a
        # 2x2 grid, visibly collapse into a single square (the sum), and that
        # sum square then fills the output cell.
        target = out[p][q][k]
        result = blend(patch_avg_color(p, q), hue)
        copies = VGroup(*[
            filt[k][r][s][0].copy()
            .set_fill(blend(input_base_color(p + r, q + s), hue), opacity=0.95)
            .set_z_index(K + 2)
            for r in range(R) for s in range(S)
        ])
        staging = target.get_center() + LEFT * 1.1  # hold the grid beside the cell
        self.play(
            copies.animate.move_to(staging).scale(0.5),
            FadeOut(lines),
            run_time=rt,
        )
        # sum: the four product squares merge into one
        sum_sq = Square(side_length=copies[0].width)
        sum_sq.set_fill(shade(result, k, K), opacity=1.0)
        sum_sq.set_stroke(WHITE, width=1.2)
        sum_sq.move_to(staging).set_z_index(K + 2)
        self.play(
            *[c.animate.move_to(staging) for c in copies],
            FadeIn(sum_sq, scale=0.6),
            run_time=0.6 if not quick else 0.35,
        )
        self.remove(copies)
        # the summed value drops into the output cell (at the card's own depth,
        # so landing on the back card doesn't cover the front one)
        sum_sq.set_z_index(target.z_index)
        self.play(
            sum_sq.animate.move_to(target.get_center()).match_width(target),
            run_time=0.4,
        )
        target.set_fill(shade(result, k, K), opacity=1.0)
        self.remove(sum_sq)

    # ============================================================= PART 2
    def part2_gemm(self):
        title = Text("Same work as an implicit GEMM:  A · B = C",
                     weight=BOLD).scale(0.6).to_edge(UP, buff=0.3)
        self.play(FadeIn(title, shift=DOWN * 0.2))

        # --- A[M, L]: each row = one output pixel's flattened receptive field ---
        def a_color(m, l):
            p, q = divmod(m, Q)
            rs, c = divmod(l, C)
            r, s = divmod(rs, S)
            return shade(input_base_color(p + r, q + s), c), 1.0

        # --- B[L, K]: each column = one flattened filter ---
        def b_color(l, k):
            _, c = divmod(l, C)
            return shade(FILTER_HUES[k], c), 1.0

        # --- C[M, K]: arrives already filled — it morphs from the output tensor ---
        def c_color(m, k):
            p, q = divmod(m, Q)
            return shade(blend(patch_avg_color(p, q), FILTER_HUES[k]), k, K), 1.0

        A, Acell = matrix_grid(M, L, a_color, side=0.17, gap=0.035)
        B, Bcell = matrix_grid(L, K, b_color, side=0.17, gap=0.035)
        Cmat, Ccell = matrix_grid(M, K, c_color, side=0.17, gap=0.035)

        Cmat.move_to([2.9, -1.35, 0])
        A.next_to(Cmat, LEFT, buff=1.6)
        B.next_to(Cmat, UP, buff=0.45)
        # align rows (A<->C) and cols (B<->C) precisely so the sweep "meets" at C
        A.align_to(Cmat, UP)
        B.align_to(Cmat, LEFT)

        # axis braces / labels
        brM = Brace(A, LEFT, buff=0.1)
        lM = brM.get_tex(r"M = P\cdot Q").scale(0.7)
        brL = Brace(A, UP, buff=0.1)
        lL = brL.get_tex(r"R\cdot S\cdot C").scale(0.7)
        brN = Brace(B, UP, buff=0.08)
        lN = brN.get_tex(r"N = K").scale(0.7)
        labA = Text("A  (input, implicit)").scale(0.32).next_to(A, DOWN, buff=0.25)
        labB = Text("B  (kernel)").scale(0.32).next_to(B, RIGHT, buff=0.25)
        labC = Text("C  (output)").scale(0.32).next_to(Cmat, DOWN, buff=0.25)

        # ---- morph the spatial tensors into their matrices ----
        # (kernel first: the filters sit over the area where A lands)
        inp, filt, out = self.inp, self.filt, self.out
        # kernel -> B: each filter unrolls into one column.
        # filter 0 is shown in detail (each spatial cell's C channel cards fly
        # into the column together, one triplet at a time, with the source cell
        # highlighted); the rest fast-forward.
        for k in range(K):
            if k == 0:
                for rs in range(R * S):
                    r, s = divmod(rs, S)
                    src = filt[k][r][s]
                    hl = SurroundingRectangle(src, color=WHITE, buff=0.04)
                    hl.set_z_index(C + 2)
                    self.play(Create(hl), run_time=0.2)
                    self.play(
                        *[ReplacementTransform(src[c], Bcell[rs * C + c][k])
                          for c in range(C)],
                        FadeOut(hl),
                        run_time=0.55,
                    )
            else:
                self.play(
                    LaggedStart(*[
                        ReplacementTransform(filt[k][r][s][c], Bcell[(r * S + s) * C + c][k])
                        for r in range(R) for s in range(S) for c in range(C)
                    ], lag_ratio=0.03),
                    run_time=0.6,
                )
        # input -> A: each output pixel's receptive field unrolls into one row.
        # Input cards are *copied* (each is reused by several rows — implicit!).
        # The yellow window returns to show which patch feeds each row; the
        # first two rows form one input pixel (all C channel cards) at a time.
        def patch_rect(p, q):
            stacks = VGroup(*[inp[p + r][q + s] for r in range(R) for s in range(S)])
            rect = SurroundingRectangle(stacks, color=WINDOW_COLOR, buff=0.06)
            rect.set_z_index(C + 1)
            return rect

        window = patch_rect(0, 0)
        self.play(Create(window))
        for m in range(M):
            p, q = divmod(m, Q)
            if m > 0:
                self.play(window.animate.move_to(patch_rect(p, q).get_center()),
                          run_time=0.4 if m <= 1 else 0.25)
            if m < 2:
                # one pixel at a time: its C channel cards fly into the row together
                for rs in range(R * S):
                    r, s = divmod(rs, S)
                    src = inp[p + r][q + s]
                    hl = SurroundingRectangle(src, color=WHITE, buff=0.04)
                    hl.set_z_index(C + 2)
                    self.play(Create(hl), run_time=0.2)
                    self.play(
                        *[TransformFromCopy(src[c], Acell[m][rs * C + c])
                          for c in range(C)],
                        FadeOut(hl),
                        run_time=0.55,
                    )
            else:
                # fast-forward: the whole row unrolls at once
                self.play(
                    LaggedStart(*[
                        TransformFromCopy(inp[p + r][q + s][c], Acell[m][(r * S + s) * C + c])
                        for r in range(R) for s in range(S) for c in range(C)
                    ], lag_ratio=0.03),
                    run_time=0.45,
                )
        self.play(FadeOut(window), FadeOut(self.inp_grid), run_time=0.5)
        # output -> C: staged so the K axis reads clearly. Channel 0 arrives one
        # spatial row at a time; then all remaining channels finish at once.
        for p in range(P):
            self.play(
                LaggedStart(*[
                    ReplacementTransform(out[p][q][0], Ccell[p * Q + q][0])
                    for q in range(Q)
                ], lag_ratio=0.1),
                run_time=0.7,
            )
        self.play(
            LaggedStart(*[
                ReplacementTransform(out[p][q][k], Ccell[p * Q + q][k])
                for p in range(P) for q in range(Q) for k in range(1, K)
            ], lag_ratio=0.05),
            run_time=1.0,
        )

        self.play(
            LaggedStart(
                GrowFromCenter(brM), FadeIn(lM),
                GrowFromCenter(brL), FadeIn(lL),
                GrowFromCenter(brN), FadeIn(lN),
                Write(labA), Write(labB), Write(labC),
                lag_ratio=0.12,
            )
        )
        self.wait(0.4)

        # ---- dot-product sweep: A row 0 . B col k -> C[0,k] ----
        def row_box(cells, m):
            return SurroundingRectangle(VGroup(*cells[m]), color=WINDOW_COLOR, buff=0.02)

        def col_box(cells, k, nrows):
            return SurroundingRectangle(VGroup(*[cells[r][k] for r in range(nrows)]),
                                        color=WINDOW_COLOR, buff=0.02)

        arow = row_box(Acell, 0)
        self.play(Create(arow))
        for k in range(K):
            bcol = col_box(Bcell, k, L)
            self.play(Create(bcol), run_time=0.5)
            # sweep the shared reduction axis, channels grouped in threes
            cursor_a = SurroundingRectangle(Acell[0][0], color=WHITE, buff=0.01)
            cursor_b = SurroundingRectangle(Bcell[0][k], color=WHITE, buff=0.01)
            self.play(FadeIn(cursor_a), FadeIn(cursor_b), run_time=0.2)
            for l in range(L):
                anims = [
                    cursor_a.animate.move_to(Acell[0][l].get_center()),
                    cursor_b.animate.move_to(Bcell[l][k].get_center()),
                ]
                # pulse every channel-group boundary to echo the C reduction
                if l % C == 0:
                    anims += [Indicate(Acell[0][l], color=WINDOW_COLOR, scale_factor=1.4),
                              Indicate(Bcell[l][k], color=WINDOW_COLOR, scale_factor=1.4)]
                self.play(*anims, run_time=0.09)
            self.play(
                Indicate(Ccell[0][k], color=WINDOW_COLOR, scale_factor=1.6),
                FadeOut(cursor_a), FadeOut(cursor_b), FadeOut(bcol),
                run_time=0.5,
            )
        self.play(FadeOut(arow))
        self.wait(0.3)

        # ---- the rest of C follows the same row-by-column pattern ----
        pulses = [
            Indicate(Ccell[m][k], color=WINDOW_COLOR, scale_factor=1.4)
            for m in range(1, M) for k in range(K)
        ]
        self.play(LaggedStart(*pulses, lag_ratio=0.04), run_time=2.0)
        self.wait(0.4)

        # ---- "implicit" callout: one input element reused across A rows ----
        # input pixel (0,1) c=0 is (r=0,s=1) of out-pixel(0,0)  -> row 0, col 3
        # and                       (r=0,s=0) of out-pixel(0,1)  -> row 1, col 0.
        e1, e2 = Acell[0][3], Acell[1][0]
        note = Text("A is never stored — each input is re-read\nfrom overlapping fields (implicit)",
                    line_spacing=0.8).scale(0.34)
        note.to_edge(DOWN, buff=0.2)
        link = CurvedArrow(e1.get_center(), e2.get_center(), color=WINDOW_COLOR,
                           radius=0.5, stroke_width=3, tip_length=0.12)
        self.play(
            Indicate(e1, color=WINDOW_COLOR, scale_factor=1.6),
            Indicate(e2, color=WINDOW_COLOR, scale_factor=1.6),
            Create(link),
            FadeIn(note, shift=UP * 0.2),
        )
        self.wait(2.0)

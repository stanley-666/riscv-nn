# FFT 完整效能重算報告

本報告統一整理 1,024-point、64-batch FFT 在 scalar CPU、RVV 與
Gemmini 上的最新實測結果。所有倍數均由 README 內的原始 cycle
重新相除，未沿用舊報告中的倍數。

## 比較口徑

- CPU 與 RVV FPGA 時脈為 50 MHz，`time (ms) = cycles / 50,000`。
- Gemmini FPGA 時脈為 30 MHz，`time (ms) = cycles / 30,000`。
- `FFT cycles` 是 bit reversal 加 butterfly，不含 input layout 與一次性的
  twiddle-plan 建立。
- 每個 RVV configuration 都取該 layout 中實測最快的 LMUL/fusion variant。
- CPU/RVV 可直接比較 cycle；Gemmini 時脈不同，跨架構倍數以毫秒計算。
- 最新 FP32/int8 radix-2 結果皆通過 0 / 65,536 mismatch。

## 跨架構總結

| Precision | 架構與最佳設定 | FFT cycles | 時脈 | FFT time | 相對同精度 CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| FP32 | Scalar CPU | 12,760,529 | 50 MHz | 255.21 ms | 1.00x |
| FP32 | RVV LGVV512D128, bin-major m4 fused | **926,835** | 50 MHz | **18.54 ms** | **13.77x faster** |
| int8 | Scalar CPU baseline | 13,801,835 | 50 MHz | 276.04 ms | 1.00x |
| int8 | RVV LGVV512D128, bin-major m4 | **1,074,399** | 50 MHz | **21.49 ms** | **12.85x faster** |
| int8 | Gemmini WS baseline | 19,073,443 | 30 MHz | 635.78 ms | **2.30x slower** |

Gemmini WS baseline 相對最快 int8 RVV 慢 **29.59x**（以時間比較）。
Gemmini 的 512,544-cycle twiddle plan 未計入上述時間；即使排除 plan，
Gemmini 仍慢於 scalar CPU。

## RVV 全部 FPGA 結果

`layout penalty = best batch-major cycles / best bin-major cycles`。兩種
layout 的 `FFT cycles` 均不含 input layout conversion。

### FP32

| Configuration | 最佳 bin variant | Bin cycles | Bin ms | 最佳 batch variant | Batch cycles | Batch ms | Layout penalty |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| GENV128D64 | m4 fused | 1,933,799 | 38.68 | m2 | 12,837,967 | 256.76 | 6.64x |
| GENV128D128 | m8 | 1,169,218 | 23.38 | m4 | 12,368,468 | 247.37 | 10.58x |
| GENV256D64 | m4 fused | 1,774,230 | 35.48 | m2 | 12,238,898 | 244.78 | 6.90x |
| GENV256D128 | m8 | 1,033,571 | 20.67 | m2 | 11,686,343 | 233.73 | 11.31x |
| GENV512D64 | m4 fused | 1,687,393 | 33.75 | m2 | 12,107,773 | 242.16 | 7.18x |
| GENV512D128 | m4 fused | 927,136 | 18.54 | m2 | 11,435,252 | 228.71 | 12.33x |
| LGVV128D128 | m8 | 1,125,523 | 22.51 | m2 | 11,867,203 | 237.34 | 10.54x |
| LGVV256D128 | m4 fused | 1,046,752 | 20.94 | m2 | 11,257,738 | 225.15 | 10.75x |
| LGVV512D128 | m4 fused | **926,835** | **18.54** | m2 | **11,022,607** | **220.45** | 11.89x |

### int8 Q7

| Configuration | 最佳 bin variant | Bin cycles | Bin ms | 最佳 batch variant | Batch cycles | Batch ms | Layout penalty |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| GENV128D64 | m8 | 2,320,895 | 46.42 | m4 | 9,350,701 | 187.01 | 4.03x |
| GENV128D128 | m8 | 1,418,607 | 28.37 | m4 | 8,485,372 | 169.71 | 5.98x |
| GENV256D64 | m4 | 2,151,407 | 43.03 | m4 | 9,112,563 | 182.25 | 4.24x |
| GENV256D128 | m8 | 1,178,280 | 23.57 | m4 | 8,219,844 | 164.40 | 6.98x |
| GENV512D64 | m4 | 2,097,075 | 41.94 | m4 | 9,039,170 | 180.78 | 4.31x |
| GENV512D128 | m4 | 1,079,519 | 21.59 | m4 | 8,129,785 | 162.60 | 7.53x |
| LGVV128D128 | m4 | 1,374,580 | 27.49 | m4 | 8,215,474 | 164.31 | 5.98x |
| LGVV256D128 | m4 | 1,150,522 | 23.01 | m4 | 8,008,818 | 160.18 | 6.96x |
| LGVV512D128 | m4 | **1,074,399** | **21.49** | m4 | **7,918,006** | **158.36** | 7.37x |

## 精度影響

| Configuration | Bin int8 / FP32 | Batch FP32 / int8 |
| --- | ---: | ---: |
| GENV128D64 | 1.200x slower | 1.373x faster |
| GENV128D128 | 1.213x slower | 1.458x faster |
| GENV256D64 | 1.213x slower | 1.343x faster |
| GENV256D128 | 1.140x slower | 1.422x faster |
| GENV512D64 | 1.243x slower | 1.339x faster |
| GENV512D128 | 1.164x slower | 1.407x faster |
| LGVV128D128 | 1.221x slower | 1.444x faster |
| LGVV256D128 | 1.099x slower | 1.406x faster |
| LGVV512D128 | 1.159x slower | 1.392x faster |

bin-major 已讓 lanes 跨 batch，FP32 可使用規則、連續的 vector access；
int8 complex multiply 卻仍須 widen 到 int16/int32，再做 rounding、
narrowing 與 saturation，因此 int8 bin-major 比 FP32 慢 1.10x–1.24x。

batch-major 的 lanes 跨 bin，layout 本身不適合 radix-2 的 stage stride，
但 e8 可容納更多 lanes，因此 int8 比同 layout FP32 快 1.34x–1.46x。
這不代表 batch-major 較好：int8 batch-major 仍比其 bin-major 慢
4.03x–7.53x。Scalar CPU 的 int8 也比 FP32 慢 **1.082x**。

## VLEN、DLEN 與 LGVV/GENV

### VLEN scaling（GENV、D128、最佳 bin-major）

| Precision | VLEN 128 | VLEN 256 | VLEN 512 | 128→256 | 256→512 | 128→512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1,169,218 | 1,033,571 | 927,136 | 1.131x | 1.115x | 1.261x |
| int8 | 1,418,607 | 1,178,280 | 1,079,519 | 1.204x | 1.091x | 1.314x |

VLEN 加倍沒有得到 2x；stage 的有效 VL、loop/control、load/store、bit
reversal、短尾向量與 register pressure 都限制 scaling。

### DLEN 64→128 speedup（GENV、相同 VLEN）

| VLEN | FP32 | int8 |
| ---: | ---: | ---: |
| 128 | 1.654x | 1.636x |
| 256 | 1.717x | 1.826x |
| 512 | 1.820x | 1.943x |

DLEN 對 throughput 的影響比單純增加 VLEN 更接近線性。

### LGVV 相對 GENV（D128、相同 VLEN）

大於 1 代表 LGVV 較快；小於 1 代表 GENV 較快。

| VLEN | FP32 GENV/LGVV | int8 GENV/LGVV |
| ---: | ---: | ---: |
| 128 | 1.039x | 1.032x |
| 256 | 0.987x | 1.024x |
| 512 | 1.000x | 1.005x |

差距只有約 -1.3% 到 +3.9%，遠小於 layout 與 DLEN 的影響。

## LMUL、fusion 與 register pressure

- FP32 D64 多由 m4 fused 勝出；D128 在較小 VLEN 有時由 m8 勝出，
  VLEN=512 則回到 m4 fused。
- int8 的較小組態可由 m8 勝出，但 VLEN=512 與所有 LGVV 多由
  baseline m4 勝出。
- fusion 減少 loop/stage overhead，卻延長 live range 並增加 register
  group 使用量；spill/reload 成本較大時，fusion 反而較慢。
- FP32 VLEN=512 m8 fused butterfly 為 5,265,212 cycles（GENV）與
  5,316,059 cycles（LGVV），分別是最佳 m4 fused 的 6.81x、6.88x。
  這是明顯的 register-pressure cliff。
- int8 widening 同時占用 e8/e16/e32 register groups，所以 m8 fusion
  更容易耗盡可用 groups。

反組譯層級分析見
[`RVV_REGISTER_PRESSURE_ANALYSIS.md`](RVV_REGISTER_PRESSURE_ANALYSIS.md)。

## int8 mixed-radix 4/16/16

| Configuration | Cycles | 50 MHz time | 相對最佳 radix-2 batch-major |
| --- | ---: | ---: | ---: |
| GENV128D64 | 15,104,587 | 302.09 ms | 1.62x slower |
| GENV128D128 | 8,342,603 | 166.85 ms | 1.02x faster |
| GENV256D64 | 14,438,567 | 288.77 ms | 1.58x slower |
| GENV256D128 | 7,579,810 | 151.60 ms | 1.08x faster |
| GENV512D64 | 16,093,798 | 321.88 ms | 1.78x slower |
| GENV512D128 | 8,318,110 | 166.36 ms | 1.02x slower |
| LGVV128D128 | 8,324,841 | 166.50 ms | 1.01x slower |
| LGVV256D128 | 7,579,625 | 151.59 ms | 1.06x faster |
| LGVV512D128 | 8,316,414 | 166.33 ms | 1.05x slower |

全域最佳 mixed-radix 仍比全域最佳 bin-major radix-2 慢 **7.05x**。
減少 stage 數不足以抵消 dense radix transform 的額外 MAC、重排與
register pressure。

## 為何 RVV 快、Gemmini 慢

RVV bin-major 直接把同一 bin 的 64 batches 放成連續資料，每個 stage
可做 vector complex multiply/add/sub，沒有乘上結構零，也不需建立
dense matrix。

Gemmini 是 dense GEMM systolic array。FFT stage 雖可表示成矩陣，但矩陣
高度 sparse；即使先把 butterfly 排成大矩陣：

- 每個 stage 的 pairing/twiddle 不同，仍需 stage 間同步與重排；
- radix-2 的有效 dot product 太短，難以長時間填滿 systolic array；
- WS 可重用 weights，但 twiddle/connection pattern 每 stage 改變，
  reuse 不像 CNN convolution；
- DMA、scratchpad packing、transpose 與 CPU/Gemmini 邊界仍有成本；
- Q7 add/sub、RNU、scaling、saturation 留在 CPU 會增加往返；用額外
  Gemmini matmul 模擬，現有 native-scaling Spike 實測也更慢。

因此 Gemmini 635.78 ms 比 CPU 276.04 ms 慢 2.30x 是合理的 mapping
結果；不是 GEMM 單元本身較慢，而是 radix-2 FFT 的有效 dense 工作量
太少，資料移動與控制成本太高。

## 來源與有效性

- FP32 RVV：[`../fft_batched/README.md`](../fft_batched/README.md)
- int8 RVV：[`../fft_batched_int8/README.md`](../fft_batched_int8/README.md)
- FP32 CPU：[`../fft_cpu/README.md`](../fft_cpu/README.md)
- int8 CPU：[`../fft_cpu_int8/README.md`](../fft_cpu_int8/README.md)
- int8 Gemmini WS：
  [`../fft_batched_int8_gemmini/README.md`](../fft_batched_int8_gemmini/README.md)

本報告不使用標示為 archived、pre-fusion 或待重跑的舊數據。Gemmini
stage-fused/native-scaling 尚無可取代 baseline 的 FPGA 完整數據，因此
未混入硬體排名。

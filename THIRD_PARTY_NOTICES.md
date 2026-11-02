# Third-Party Notices

This repository includes code adapted from the third-party projects listed below. Each part remains under its
original license, reproduced or referenced here. Everything else is covered by the repository's `LICENSE`
(CC BY-NC-SA 4.0). Python packages that are only installed as dependencies (see `requirements.txt`) are not listed.

| # | Project | License | Used in |
|---|---------|---------|---------|
| 1 | [coco-caption](https://github.com/tylin/coco-caption) | BSD 2-Clause | `score/cider.py`, `score/cider_scorer.py`, `score/rouge.py` |
| 2 | [Hugging Face Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | `module/EarthVLNet.py`, `module/base_modules.py` |
| 3 | [Hugging Face PEFT](https://github.com/huggingface/peft) | Apache-2.0 | `module/EarthVLNet.py` |
| 4 | [LLaVA](https://github.com/haotian-liu/LLaVA) | Apache-2.0 | `module/base_modules.py` |
| 5 | [vit-pytorch](https://github.com/lucidrains/vit-pytorch) | MIT | `module/base_modules.py` |
| 6 | [MCAN-VQA](https://github.com/MILVLG/mcan-vqa) | Apache-2.0 | `module/base_modules.py` |
| 7 | [torchvision](https://github.com/pytorch/vision) | BSD 3-Clause | `module/base_modules.py` |
| 8 | [Albumentations](https://github.com/albumentations-team/albumentations) | MIT | `module/data_augs.py` |
| 9 | [OpenMMLab MMClassification / MMCV](https://github.com/open-mmlab/mmpretrain) | Apache-2.0 | `module/seg/convnext.py` |
| 10 | [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) | MIT | `module/seg/base.py` |
| 11 | [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | MIT | `module/seg/convnext.py` |
| 12 | [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | `module/seg/convnext.py` |

## 1. coco-caption (Microsoft COCO Caption Evaluation)

- Source: <https://github.com/tylin/coco-caption> (`pycocoevalcap/cider/cider.py`, `pycocoevalcap/cider/cider_scorer.py`,
  `pycocoevalcap/rouge/rouge.py`).
- Used in: `score/cider.py`, `score/cider_scorer.py`, `score/rouge.py` (CIDEr and ROUGE-L for open-ended answers).
- Changes: `compute_score` of `Cider` and `Rouge` takes lists (one entry per sample) instead of dicts keyed by image
  id; Python 3 imports.

```text
Copyright (c) 2015, Xinlei Chen, Hao Fang, Tsung-Yi Lin, and Ramakrishna Vedantam
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
```

## 2. Hugging Face Transformers

- Source: <https://github.com/huggingface/transformers>, `src/transformers/models/llava/modeling_llava.py` and
  `src/transformers/models/llava_next/modeling_llava_next.py` (v4.40). Copyright 2023-2024 The HuggingFace Inc. team.
- Used in:
  - `module/EarthVLNet.py`: `EarthVLNet._merge_input_ids_with_image_features` and `EarthVLNet.llm_forward` (merging
    the visual tokens into the text embeddings at the `<image>` token, the attention mask for cached generation and
    the shifted language-modelling loss) are adapted from `LlavaForConditionalGeneration` /
    `LlavaNextForConditionalGeneration`.
  - `module/base_modules.py`: `MMProjector` follows `LlavaMultiModalProjector`.
- License: Apache License 2.0 (full text at the end of this file).

## 3. Hugging Face PEFT

- Source: <https://github.com/huggingface/peft> (`PeftModel.get_nb_trainable_parameters` and
  `PeftModel.print_trainable_parameters`). Copyright 2023-present the HuggingFace Inc. team.
- Used in: `module/EarthVLNet.py` (`EarthVLNet.get_nb_trainable_parameters`, `EarthVLNet.print_trainable_parameters`).
- License: Apache License 2.0 (full text at the end of this file).

## 4. LLaVA

- Source: <https://github.com/haotian-liu/LLaVA> (Haotian Liu et al.; `llava/train/train.py`).
- Used in: `module/base_modules.py` (`rank0_print` and `find_all_linear_names`).
- License: Apache License 2.0 (full text at the end of this file).

## 5. vit-pytorch

- Source: <https://github.com/lucidrains/vit-pytorch> (`vit_pytorch/vit.py`).
- Used in: `module/base_modules.py` (`FeedForward`, `Attention` and the layer stack of `Transformer`).

```text
MIT License

Copyright (c) 2020 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 6. MCAN-VQA (Deep Modular Co-Attention Networks)

- Source: <https://github.com/MILVLG/mcan-vqa> (Yuhao Cui, MILVLG; `core/model/net_utils.py`, `core/model/net.py`).
  The repository is licensed under the Apache License 2.0; the headers of its source files state the MIT License.
- Used in: `module/base_modules.py` (`FC`, `MLP`, `AttFlat`).
- License: Apache License 2.0 (full text at the end of this file).

## 7. torchvision

- Source: <https://github.com/pytorch/vision> (`torchvision/models/convnext.py`).
- Used in: `module/base_modules.py` (`LayerNorm2d`).

```text
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 8. Albumentations

- Source: <https://github.com/albumentations-team/albumentations> (v1.4.3).
- Used in: `module/data_augs.py`. `VQAHorizontalFlip`, `VQAVerticalFlip` and `VQARandomRotate90` subclass the
  albumentations transforms, and their `apply` methods follow the albumentations implementations.

```text
MIT License

Copyright (c) 2017 Vladimir Iglovikov, Alexander Buslaev, Alexander Parinov,

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 9. OpenMMLab MMClassification / MMCV

- Source: <https://github.com/open-mmlab/mmpretrain> (MMClassification 0.x, `mmcls/models/backbones/convnext.py`)
  and <https://github.com/open-mmlab/mmcv> (`DropPath`, the weight initialisers), Copyright (c) OpenMMLab.
- Used in: `module/seg/convnext.py` (the ConvNeXt backbone of the stage-1 segmentation network, rewritten without
  mmcv; the OpenMMLab copyright header is kept in the file). The MMClassification ConvNeXt is itself modified from the
  official ConvNeXt code (item 11) and from timm (item 12).
- License: Apache License 2.0 (full text at the end of this file).

## 10. maskrcnn-benchmark

- Source: <https://github.com/facebookresearch/maskrcnn-benchmark> (`maskrcnn_benchmark/modeling/backbone/fpn.py`,
  `maskrcnn_benchmark/modeling/make_layers.py`).
- Used in: `module/seg/base.py` (`FPN`, `LastLevelMaxPool`, `LastLevelP6P7`, and `conv_with_kaiming_uniform`, here with
  batch normalization instead of group normalization).

```text
MIT License

Copyright (c) 2018 Facebook

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 11. ConvNeXt

- Source: <https://github.com/facebookresearch/ConvNeXt> (`models/convnext.py`), through the MMClassification port
  (item 9).
- Used in: `module/seg/convnext.py` (`ConvNeXtBlock`, `ConvNeXt`).

```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 12. PyTorch Image Models (timm)

- Source: <https://github.com/huggingface/pytorch-image-models> (`timm/models/convnext.py`), through the
  MMClassification port (item 9). Copyright 2019 Ross Wightman.
- Used in: `module/seg/convnext.py` (`ConvNeXtBlock`, `ConvNeXt`).
- License: Apache License 2.0 (full text at the end of this file).

## Apache License 2.0

Full text of the license of items 2, 3, 4, 6, 9 and 12 (also at <https://www.apache.org/licenses/LICENSE-2.0>).

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

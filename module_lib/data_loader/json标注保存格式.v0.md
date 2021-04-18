Tips:

图片在windows属性上 的显示为 [宽度像素 x 高度像素]

在 cv2 的照片属性中， img.shape 为 [高度像素 x 宽度像素 x 通道数]



json标注保存格式.v0

```json
{

	"version": "zen_v1.0.0",

	"flags":{},

	"imagePath": $Path,

	"imageData": null,

	"imageHeight": $pixelHeight,

	"imageWidth": $pixelWidth,

	"shape": [{label_1}, 
              {label_2}, 
              ...],

}


```



label可包含：

- 点 - point
- 矩形 - retangle
- 多边形 - polygen
- 线 - line
- 不闭环线段组合 - linestrip
- 圆形 - circle



label的标准格式为：

```json
{

	"label": $label_name (即标记的名字)，

	"points": [依据不同类型的label],

	"group_id": null,

	"shape": 只能从[point, retangle, polygen, line, linestrip, circle]中选择,

}
```



label-点(point)示例：

```json
{
      "label": "point_sample",
      "points": [
        [
          1471.2222222222222,
          169.55555555555554
        ]
      ],
      "group_id": null,
      "shape_type": "point",
      "flags": {}
}
```



label-矩形(retangle)示例：

```json
{
      "label": "Retangle_sample",
      "points": [
        [
          300.85185185185185,
          320.48148148148147
        ],
        [
          560.1111111111111,
          553.8148148148148
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
}
```



label-多边形矩形(polygen)示例：

```json
{
      "label": "polygen_sample",
      "points": [
        [
          1582.3333333333333,
          711.2222222222222
        ],
        [
          1434.185185185185,
          815.8518518518518
        ],
        [
          1535.111111111111,
          963.0740740740739
        ],
        [
          1720.296296296296,
          935.2962962962962
        ],
        [
          1561.037037037037,
          861.2222222222222
        ]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
}
```



label-线(line)示例：

```json
{
      "label": "line_sample",
      "points": [
        [
          563.8148148148148,
          118.62962962962962
        ],
        [
          990.6666666666666,
          177.88888888888889
        ]
      ],
      "group_id": null,
      "shape_type": "line",
      "flags": {}
}
```



label-不闭环线段组合(linestrip)示例：

```json
{
      "label": "linestrip_sample",
      "points": [
        [
          1463.8148148148148,
          247.33333333333331
        ],
        [
          1400.8518518518517,
          422.3333333333333
        ],
        [
          1630.4814814814813,
          419.55555555555554
        ],
        [
          1532.3333333333333,
          553.8148148148148
        ],
        [
          1428.6296296296296,
          514.0
        ],
        [
          1465.6666666666665,
          514.9259259259259
        ]
      ],
      "group_id": null,
      "shape_type": "linestrip",
      "flags": {}
}
```



label-圆形(circle)示例：

```json
{
      "label": "circle_sample",
      "points": [
        [
          445.29629629629625,
          839.9259259259259
        ],
        [
          316.59259259259255,
          861.2222222222222
        ]
      ],
      "group_id": null,
      "shape_type": "circle",
      "flags": {}
}
```


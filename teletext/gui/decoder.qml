import QtQuick 2.12
import QtGraphicalEffects 1.12


Rectangle {
    property var borderSize: ttfonts[0][0].pixelSize / 2
    width: teletext.width + borderSize * 2
    height: teletext.height + borderSize * 2
    border.width: borderSize
    border.color: "black"
    color: "black"
    GridView {
        id: teletext
        width: (40 * 8 * ttfonts[0][0].pixelSize / 10)
        height: ttfonts[0][0].pixelSize * 25
        cellHeight: ttfonts[0][0].pixelSize
        cellWidth: 8 * ttfonts[0][0].pixelSize / 10
        x: parent.borderSize
        y: parent.borderSize
        interactive: false
        clip: true
        model: ttmodel
        delegate: Rectangle {
            color: display.bg
            Text {
                x: -0.4
                color: display.fg
                text: display.text
                font: ttfonts[display.width-1][display.height-1]
            }
            height: display.height * ttfonts[0][0].pixelSize
            width: display.width * 8 * ttfonts[0][0].pixelSize / 10
            clip: true
            visible: display.visible
        }
        layer.enabled: tteffect && (ttfonts[0][0].pixelSize > 10)
        layer.effect: ShaderEffect {
            fragmentShader: "
                    uniform lowp sampler2D source;
                    uniform lowp float qt_Opacity;
                    varying highp vec2 qt_TexCoord0;
                    varying lowp vec3 qt_FragCoord0;
                    void main() {
                        lowp vec4 tex = texture2D(source, qt_TexCoord0);
                        gl_FragColor = (int(gl_FragCoord.y)%(" + (ttfonts[0][0].pixelSize / 10) + ")) == 1 ? tex : tex*0.5;
                    }
                "
        }
    }
    layer.enabled: tteffect && (ttfonts[0].pixelSize > 10)
    layer.effect: FastBlur {
        radius: 1.75;
    }
}



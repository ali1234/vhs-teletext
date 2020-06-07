import QtQuick 2.9
import QtGraphicalEffects 1.12


Rectangle {
    property var borderSize: ttfont.pixelSize
    width: teletext.width + borderSize * 2
    height: teletext.height + borderSize * 2
    border.width: borderSize
    border.color: "black"
    color: "black"
    ListView {
        id: teletext
        width: 40 * 7.68 * ttfont.pixelSize / 10
        height: ttfont.pixelSize * 25
        x: parent.borderSize
        y: parent.borderSize
        interactive: false
        model: ttmodel
        delegate: Text {
            color: "white"
            textFormat: Text.RichText
            text: display
            font: ttfont
            height: ttfont.pixelSize
            clip: true
            layer.enabled: tteffect && (ttfont.pixelSize > 10)
            layer.effect: ShaderEffect {
                fragmentShader: "
                    uniform lowp sampler2D source;
                    uniform lowp float qt_Opacity;
                    varying highp vec2 qt_TexCoord0;
                    void main() {
                        lowp vec4 tex = texture2D(source, qt_TexCoord0);
                        gl_FragColor = (int(qt_TexCoord0.y*2500)%10)<7 ? tex : tex*0.4;
                    }
                "
            }
        }
    }
    layer.enabled: tteffect && (ttfont.pixelSize > 10)
    layer.effect: FastBlur {
        radius: 0.75;
    }
}



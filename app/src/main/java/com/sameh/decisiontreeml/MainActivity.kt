package com.sameh.decisiontreeml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.doOnTextChanged
import com.google.android.material.textfield.TextInputLayout
import com.sameh.decisiontreeml.databinding.ActivityMainBinding
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    private var _binding: ActivityMainBinding? = null
    private val binding get() = _binding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        _binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setActions()
    }

    override fun onDestroy() {
        super.onDestroy()
        _binding = null
    }

    private fun setActions() {
        binding.apply {
            layoutSepalLength.removeErrorTextWhenTextChanges()
            layoutSepalWidth.removeErrorTextWhenTextChanges()
            layoutPetalLength.removeErrorTextWhenTextChanges()
            layoutPetalWidth.removeErrorTextWhenTextChanges()

            btnPredict.setOnClickListener {
                checkInputs()
            }
        }
    }

    private fun checkInputs() {
        binding.apply {
            val sepalLength = etSepalLength.text.toString()
            val sepalWidth = etSepalWidth.text.toString()
            val patelLength = etPetalLength.text.toString()
            val patelWidth = etPetalWidth.text.toString()

            if (sepalLength.isEmpty())
                layoutSepalLength.error = "Required"
            if (sepalWidth.isEmpty())
                layoutSepalWidth.error = "Required"
            if (patelLength.isEmpty())
                layoutPetalLength.error = "Required"
            if (patelWidth.isEmpty())
                layoutPetalWidth.error = "Required"

            if (sepalLength.isNotEmpty() && sepalWidth.isNotEmpty() && patelLength.isNotEmpty() && patelWidth.isNotEmpty()) {
                val irisResult = handlePrediction(
                    floatArrayOf(
                        sepalLength.toFloat(),
                        sepalWidth.toFloat(),
                        patelLength.toFloat(),
                        patelWidth.toFloat()
                    )
                )
                updateUiText(irisResult)
            }
        }
    }

    private fun updateUiText(value: String) {
        binding.apply {
            tvIrisResult.text = value

            when (value) {
                "Iris-setosa" -> {
                    tvIrisDes.text = getString(R.string.iris_setosa_description)
                }
                "Iris-versicolor" -> {
                    tvIrisDes.text = getString(R.string.iris_versicolor_description)
                }
                "Iris-virginica" -> {
                    tvIrisDes.text = getString(R.string.iris_virginica_description)
                }
            }
        }
    }

    private fun handlePrediction(features: FloatArray): String {
        val ortEnvironment = OrtEnvironment.getEnvironment()
        val ortSession = createORTSession(ortEnvironment)
        return runPrediction(features, ortSession, ortEnvironment)
    }

    private fun createORTSession(ortEnvironment: OrtEnvironment): OrtSession {
        val modelBytes = resources.openRawResource(R.raw.decision_tree_model).readBytes()
        return ortEnvironment.createSession(modelBytes)
    }

    private fun runPrediction(
        inputs: FloatArray,
        ortSession: OrtSession,
        ortEnvironment: OrtEnvironment
    ): String {
        try {
            // Get the name of the input node
            val inputName = ortSession.inputNames?.iterator()?.next()
            // Make a FloatBuffer of the inputs
            val floatBufferInputs = FloatBuffer.wrap(inputs)
            // Create input tensor with floatBufferInputs of shape (1, 4)
            val inputTensor = OnnxTensor.createTensor(
                ortEnvironment,
                floatBufferInputs,
                longArrayOf(1, inputs.size.toLong())
            )
            // Run the model
            val results = ortSession.run(mapOf(inputName to inputTensor))
            // Fetch and return the results

            return when (val outputValue = results[0].value) {
                is Array<*> -> {
                    // Convert the array of strings to a single string
                    val stringValue = outputValue.joinToString(" ")
                    stringValue.toLogD()
                    stringValue
                }

                else -> return "Unsupported output type: ${outputValue?.javaClass}"
            }
        } catch (e: Exception) {
            e.message?.toLogE()
            return e.message ?: ""
        }
    }

    private fun String.toLogD(tag: String = "debugTAG") {
        Log.d(tag, this)
    }

    private fun String.toLogE(tag: String = "debugTAG") {
        Log.e(tag, this)
    }

    private fun TextInputLayout.removeErrorTextWhenTextChanges() {
        this.editText?.doOnTextChanged { _, _, _, _ ->
            this.error = null
        }
    }
}
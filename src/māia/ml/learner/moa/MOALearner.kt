package māia.ml.learner.moa

import com.yahoo.labs.samoa.instances.*
import moa.classifiers.MultiClassClassifier
import moa.core.Example
import moa.learners.Learner
import māia.ml.dataset.DataRow
import māia.ml.dataset.DataStream
import māia.ml.dataset.WithColumnHeaders
import māia.ml.dataset.moa.dataRowToInstanceExample
import māia.ml.dataset.moa.withColumnHeadersToInstancesHeader
import māia.ml.dataset.type.Nominal
import māia.ml.dataset.util.buildRow
import māia.ml.dataset.util.ifIsPossiblyMissing
import māia.ml.dataset.view.readOnlyViewColumns
import māia.ml.learner.AbstractLearner
import māia.ml.learner.type.Classifier
import māia.ml.learner.type.LearnerType
import māia.ml.learner.type.SingleTarget
import māia.ml.learner.type.intersectionOf
import māia.util.maxWithIndex

val MOA_LEARNER_TYPE = intersectionOf(SingleTarget, Classifier)

/**
 * TODO
 */
class MOALearner(
    val source : Learner<Example<Instance>>
) : AbstractLearner<DataStream<*>>(
    MOA_LEARNER_TYPE,
    DataStream::class
) {
    init {
        if (source !is MultiClassClassifier)
            throw Exception("source is not a MultiClassClassifier")
    }

    lateinit var instancesHeader : InstancesHeader

    override fun performInitialisation(
        headers : WithColumnHeaders
    ) : Triple<WithColumnHeaders, WithColumnHeaders, LearnerType> {
        val predictInputHeaders = headers.readOnlyViewColumns { _, header -> !header.isTarget}
        val predictOutputHeaders = headers.readOnlyViewColumns { _, header -> header.isTarget}

        instancesHeader = withColumnHeadersToInstancesHeader(headers)

        source.modelContext = instancesHeader

        return Triple(
            predictInputHeaders,
            predictOutputHeaders,
            MOA_LEARNER_TYPE
        )
    }

    override fun performTrain(
        trainingDataset : DataStream<*>
    ) {
        for (row in trainingDataset.rowIterator())
            source.trainOnInstance(
                dataRowToInstanceExample(row, instancesHeader)
            )
    }

    override fun performPredict(
        row : DataRow
    ) : DataRow {
        val instance = dataRowToInstanceExample(row, instancesHeader, true)

        val votes = source.getVotesForInstance(instance)

        val prediction = votes.iterator().maxWithIndex().first

        return predictOutputHeaders.buildRow { _, header ->
            val type = header.type
            ifIsPossiblyMissing<Nominal<*>>(type) then {
                it.convertIndexToInternal(prediction)
            } otherwise {
                throw Exception("Can only predict nominal values")
            }
        }
    }

}
